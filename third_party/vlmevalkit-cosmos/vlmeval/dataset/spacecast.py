"""SpaceCast-Bench: predictive spatial reasoning over multi-view indoor images.

Reference:
  SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models
  https://arxiv.org/abs/2610.12402
  data: https://huggingface.co/datasets/hongxingli/SpaceCast-Bench (pinned revision below)
  code: https://github.com/ZJU-REAL/SpaceCast-Bench (protocol "spacecast-v1")

Local addition (Cosmos3-Reasoner eval repo; not in NVIDIA upstream).

Two keys, one per official evaluation mode:
  SpaceCastBench_CoT  -> official `--mode cot`    (brief reasoning, last line "Answer: <letter>";
                         the official main experiment, our suite default)
  SpaceCastBench      -> official `--mode direct` (answer letter only)

What is taken from the official evaluator (verbatim copy in utils/spacecast_official.py):
  * system prompt, user prompt (question, blank line, "A) ..." option lines, blank line, mode suffix,
    multi-select suffix for Attachment), message order: system, then user [prompt text, images...]
    with no per-image labels (backends.api_content);
  * image order main A, bridge views V1..Vk, main B (the parquet `images` order);
  * image preprocessing: RGB, thumbnail to longest side 1280, JPEG quality 90 (prepare_image). It is
    applied once, when the TSV is built; the JPEG bytes are written to disk unchanged and sent as is
    (CosmosReason* with img_size=-1 base64-encodes the file bytes without re-encoding);
  * answer extraction (parse_response) and per-question correctness (parse_prediction exact set match);
  * metric: per-category accuracy over the 16 "<level> / <type>" categories, macro average
    (macro_type_accuracy) is the headline.

Cache: $LMUData/SpaceCastBench.tsv + $LMUData/images/SpaceCastBench/ + SpaceCastBench.manifest.json
(HF revision, parquet sha256, TSV md5, sha256 of every image). It is built under a file lock
($LMUData/SpaceCastBench.lock) and rebuilt whenever it does not validate against the manifest.
"""
import ast
import fcntl
import hashlib
import json
import os
import os.path as osp
import re
import uuid
import warnings
from collections import defaultdict
from contextlib import contextmanager

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset
from .utils import spacecast_official as sc

HF_REPO = 'hongxingli/SpaceCast-Bench'
HF_REVISION = '45be662ca54fd6f89c25b1443a54825fb03d976e'   # 2026-10-16
HF_FILE = 'data/test-00000-of-00001.parquet'
HF_FILE_SHA256 = '27541adff7aec596d0846ccb86b5d49df01366087e2dc050834946b48c426ea2'   # LFS oid @ HF_REVISION
# sha256 of '\n'.join(sorted(question_uid)) over the 3,862 released questions
RELEASE_UIDS_SHA256 = '4ec8172316d3a0e25e49299bb0630633ce38dfb19cb2411c968add27f136dfa9'
OFFICIAL_CODE_COMMIT = '6696e5eb41371a9c8038c0d902d2aa488b800593'
FAIL_MSG = 'Failed to obtain answer via API.'
TSV_NAME = 'SpaceCastBench.tsv'   # shared by both modes
MANIFEST_NAME = 'SpaceCastBench.manifest.json'
LOCK_NAME = 'SpaceCastBench.lock'
LOCAL_PARQUET = osp.join('SpaceCast-Bench', 'hf', osp.basename(HF_FILE))   # optional pre-downloaded copy
IMG_DIR = 'SpaceCastBench'
BUILD_VERSION = 2   # bump when build_tsv output changes
LEVEL_ORDER = ('L1', 'L2', 'L3')


def _official_row(line):
    """The fields the official build_prompt / parse_response read, from one TSV row."""
    return {'question': line['question'], 'options': json.loads(line['options']),
            'multi_select': _as_bool(line['multi_select'])}


def _as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() == 'true'
    return bool(v)


# ---------------------------------------------------------------- cache build / validation

def _hash_file(path, algo, block=1 << 20):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def _tmp_path(path):
    """Per-process temp name next to `path`, so concurrent builders never share a temp file."""
    return f'{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp'


def _atomic_write(path, data):
    tmp = _tmp_path(path)
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if osp.exists(tmp):
            os.remove(tmp)


@contextmanager
def _build_lock(root):
    os.makedirs(root, exist_ok=True)
    with open(osp.join(root, LOCK_NAME), 'a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _tsv_image_paths(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t', usecols=['image_path'])
    return sorted({p for ps in df['image_path'] for p in ast.literal_eval(ps)})


def validate_cache(root):
    """None when TSV + images match the manifest (and the pinned release), else the reason."""
    tsv_path, man_path = osp.join(root, TSV_NAME), osp.join(root, MANIFEST_NAME)
    img_root = osp.join(root, 'images', IMG_DIR)
    if not osp.exists(tsv_path):
        return 'no TSV'
    if not osp.exists(man_path):
        return 'no manifest'
    try:
        with open(man_path) as f:
            man = json.load(f)
    except Exception as e:
        return f'unreadable manifest ({e})'
    for k, v in (('hf_repo', HF_REPO), ('hf_revision', HF_REVISION), ('parquet_sha256', HF_FILE_SHA256),
                 ('build_version', BUILD_VERSION)):
        if man.get(k) != v:
            return f'manifest {k}={man.get(k)!r}, expected {v!r}'
    if _hash_file(tsv_path, 'md5') != man.get('tsv_md5'):
        return 'TSV md5 differs from manifest'
    images = man.get('images') or {}
    if sorted(images) != _tsv_image_paths(tsv_path):
        return 'manifest image list differs from the images the TSV references'
    for rel, digest in images.items():
        path = osp.join(img_root, rel)
        if not osp.isfile(path):
            return f'missing image {rel}'
        if _hash_file(path, 'sha256') != digest:
            return f'image checksum differs: {rel}'
    return None


def resolve_parquet(root):
    """Pinned parquet: the local copy $LMUData/SpaceCast-Bench/hf/<file> if present, else
    hf_hub_download at HF_REVISION. Its sha256 must equal the pinned LFS oid."""
    path = osp.join(root, LOCAL_PARQUET)
    if not osp.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, HF_FILE, repo_type='dataset', revision=HF_REVISION)
    digest = _hash_file(path, 'sha256')
    if digest != HF_FILE_SHA256:
        raise ValueError(f'SpaceCast-Bench parquet {path} has sha256 {digest}, expected {HF_FILE_SHA256} '
                         f'({HF_REPO}@{HF_REVISION}); delete it to re-download')
    return path


def ensure_built(root):
    """Validate the cache, (re)building TSV + images + manifest if needed, under a file lock.
    Returns the manifest."""
    man_path = osp.join(root, MANIFEST_NAME)
    with _build_lock(root):
        reason = validate_cache(root)
        if reason is not None:
            if reason not in ('no TSV',):
                warnings.warn(f'SpaceCast-Bench cache invalid ({reason}); rebuilding')
            parquet = resolve_parquet(root)
            tsv_path = osp.join(root, TSV_NAME)
            df, images = build_tsv(parquet, tsv_path, osp.join(root, 'images', IMG_DIR))
            man = {'hf_repo': HF_REPO, 'hf_revision': HF_REVISION, 'hf_file': HF_FILE,
                   'parquet_sha256': HF_FILE_SHA256, 'parquet_path': osp.abspath(parquet),
                   'official_code_commit': OFFICIAL_CODE_COMMIT, 'build_version': BUILD_VERSION,
                   'tsv_md5': _hash_file(tsv_path, 'md5'), 'release_check': release_check(df),
                   'images': dict(sorted(images.items()))}
            _atomic_write(man_path, json.dumps(man, indent=1).encode())
            reason = validate_cache(root)
            if reason is not None:
                raise RuntimeError(f'SpaceCast-Bench build does not validate: {reason}')
        with open(man_path) as f:
            return json.load(f)


def build_tsv(parquet_path, tsv_path, img_root):
    """Convert the HF parquet to a meta-only TSV plus preprocessed JPEGs under img_root.

    Images are stored once per `image_paths` entry (dataset/scene/frame); a path that maps to two
    different preprocessed images raises. Returns (dataframe, {image rel path: sha256}).
    Call it through ensure_built (file lock + manifest).
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    records, written = [], {}
    idx = 0
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            code = sc.source_type(row)
            options = list(row['options'])
            multi = bool(row.get('multi_select', False))
            allowed = [chr(65 + i) for i in range(len(options))]
            target = sc.parse_prediction(row['answer'], allowed, multi)
            if target is None:
                raise ValueError(f"Invalid gold answer: {row['question_uid']}")
            embedded = row['images']
            paths = list(row.get('image_paths') or [])
            if not 1 <= len(embedded) <= 8 or len(paths) != len(embedded):
                raise ValueError(f"Incorrect image count: {row['question_uid']}")
            rel_paths = []
            for item, rel in zip(embedded, paths):
                if item.get('path') and item['path'] != rel:
                    raise ValueError(f"Embedded image order disagrees with references: {row['question_uid']}")
                blob = sc.prepare_image(item['bytes'])
                digest = hashlib.sha256(blob).hexdigest()
                out_rel = rel if rel.lower().endswith(('.jpg', '.jpeg')) else rel + '.jpg'
                if out_rel in written:
                    if written[out_rel] != digest:
                        raise ValueError(f'Different preprocessed images for {out_rel}')
                else:
                    out = osp.join(img_root, out_rel)
                    os.makedirs(osp.dirname(out), exist_ok=True)
                    _atomic_write(out, blob)
                    written[out_rel] = digest
                rel_paths.append(out_rel)
            rec = {
                'index': idx, 'question_uid': row['question_uid'], 'question': row['question'],
                'options': json.dumps(options, ensure_ascii=False),
                'answer': ','.join(sorted(target)), 'multi_select': multi,
                'level': row['level'], 'type': sc.TYPE_NAMES[code], 'source_type': code,
                'category': sc.category(row), 'dataset': row.get('dataset'), 'scene_id': row.get('scene_id'),
                'image_roles': json.dumps(sc.view_metadata(paths)['image_roles']),
                'image_path': str(rel_paths),
            }
            for i, opt in enumerate(options):   # letter columns, for display / generic tooling only
                rec[chr(65 + i)] = opt
            records.append(rec)
            idx += 1
    df = pd.DataFrame(records)
    check = release_check(df)
    if not check['matches_release']:
        warnings.warn(f'SpaceCast-Bench data does not match the published release: {check}')
    tmp = _tmp_path(tsv_path)
    try:
        df.to_csv(tmp, sep='\t', index=False)
        os.replace(tmp, tsv_path)
    finally:
        if osp.exists(tmp):
            os.remove(tmp)
    return df, written


def release_check(df):
    """Official data.release_check (questions, scenes, per-category counts) plus question ids:
    unique, and the sorted id set hashes to RELEASE_UIDS_SHA256."""
    counts = df['category'].value_counts().to_dict()
    uids = [str(u) for u in df['question_uid']]
    uid_sha = hashlib.sha256('\n'.join(sorted(uids)).encode()).hexdigest()
    out = {'questions': len(df), 'categories': len(counts),
           'unique_uids': len(set(uids)) == len(uids), 'uid_set_matches': uid_sha == RELEASE_UIDS_SHA256,
           'category_counts_match': counts == sc.RELEASE_CATEGORY_COUNTS}
    if 'scene_id' in df and 'dataset' in df:
        out['scenes'] = len({(d, s) for d, s in zip(df['dataset'], df['scene_id'])})
    out['matches_release'] = bool(
        len(df) == sc.RELEASE_QUESTIONS and out['unique_uids'] and out['uid_set_matches']
        and out['category_counts_match'] and out.get('scenes', sc.RELEASE_SCENES) == sc.RELEASE_SCENES)
    return out


# ---------------------------------------------------------------- scoring

_ANSWER_KEY = re.compile(r'(?:FINAL\s+)?(?:ANSWERS?|CHOICES?|OPTIONS?)(?:\s+IS|\s+ARE)?\s*[:：-]?\s*([^\r\n]*)')
_LETTER = r'[\(\[]?\s*[A-Z]\s*[\)\]]?'
_LETTER_LIST = re.compile(rf'{_LETTER}(?:\s*(?:[,;/&+|\-]|\bAND\b|\bOR\b)\s*{_LETTER})*\s*[.!]?')


def lenient_parse(raw, row, parsed):
    """Diagnostic only; scoring is unchanged. Flags answers the official parser accepts leniently.

    'out_of_range': the last "Answer:/Choice:/Option:" segment is a letter list with a letter that is
        not an option (official: "Answer: A, Z" -> A).
    'letter_from_word': the parsed letter is the first character of a word after the answer keyword
        (official: "Answer: Blue chair" -> B); 'letter_from_word_conflict' when that segment is the
        text of a different option.
    '' when neither applies or nothing was parsed.
    """
    if not isinstance(raw, str) or not parsed:
        return ''
    allowed = {chr(65 + i) for i in range(len(row['options']))}
    matches = list(_ANSWER_KEY.finditer(raw.strip().upper()))
    if not matches:
        return ''
    seg = matches[-1].group(1).strip().strip('*').strip()
    if _LETTER_LIST.fullmatch(seg):
        letters = re.findall(r'(?<![A-Z0-9])([A-Z])(?![A-Z0-9])', seg)
        return 'out_of_range' if any(x not in allowed for x in letters) else ''
    m = re.match(r'[\(\[]?\s*([A-Z])[A-Z]', seg)
    if m and len(parsed) == 1 and m.group(1) == parsed[0]:
        text = sc._normalized_option_text(seg)
        hits = [chr(65 + i) for i, o in enumerate(row['options']) if sc._normalized_option_text(o) == text]
        return 'letter_from_word_conflict' if hits and hits != [parsed[0]] else 'letter_from_word'
    return ''


def score_rows(rows):
    """Per-question correctness exactly as spacecast_eval.scoring.score.

    rows: iterable of dicts with question/options (list)/multi_select/answer/prediction, where
    prediction is the raw model text (None / NaN / FAIL_MSG = no answer).
    Returns a list of (status, parsed letters or None, correct, lenient_parse flag) tuples.
    """
    out = []
    for row in rows:
        allowed = [chr(65 + i) for i in range(len(row['options']))]
        multi = bool(row['multi_select'])
        target = sc.parse_prediction(row['answer'], allowed, multi)
        if target is None:
            raise ValueError(f"Invalid gold answer: {row.get('question_uid')}")
        raw = row.get('prediction')
        if not isinstance(raw, str) or FAIL_MSG in raw:
            # official: an API failure is status "error" and counts as wrong
            out.append(('error', None, False, ''))
            continue
        parsed = sc.parse_prediction(sc.parse_response(raw, row), allowed, multi)
        status = 'valid' if parsed is not None else 'invalid'
        letters = sorted(parsed) if parsed is not None else None
        out.append((status, letters, parsed == target, lenient_parse(raw, row, letters)))
    return out


class SpaceCastBench(ImageBaseDataset):

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'
    DATASET_MODE = {'SpaceCastBench': 'direct', 'SpaceCastBench_CoT': 'cot'}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_MODE)

    def __init__(self, dataset='SpaceCastBench', **kwargs):
        self.mode = sc.check_mode(self.DATASET_MODE[dataset])
        super().__init__(dataset=dataset, **kwargs)
        self.img_root = osp.join(LMUDataRoot(), 'images', IMG_DIR)

    def load_data(self, dataset):
        root = LMUDataRoot()
        tsv_path = osp.join(root, TSV_NAME)
        self.data_path = tsv_path
        self.manifest = ensure_built(root)
        return load(tsv_path)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        paths = line['image_path']
        paths = ast.literal_eval(paths) if isinstance(paths, str) and paths.startswith('[') else (
            paths if isinstance(paths, list) else [paths])
        prompt = sc.build_prompt(_official_row(line), self.mode)
        # Official OpenAI-compatible request: system prompt, then user content [text, image_1..image_n].
        # A role='system' item is moved into the model's system prompt by BaseAPI.generate.
        msgs = [dict(role='system', type='text', value=sc.SYSTEM_PROMPT),
                dict(type='text', value=prompt)]
        msgs.extend(dict(type='image', value=osp.join(self.img_root, p)) for p in paths)
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        rows = []
        for _, r in data.iterrows():
            pred = r.get('prediction')
            rows.append({'question_uid': r['question_uid'], 'question': r['question'],
                         'options': json.loads(r['options']), 'multi_select': _as_bool(r['multi_select']),
                         'answer': str(r['answer']), 'prediction': pred if isinstance(pred, str) else None})
        res = score_rows(rows)
        data['status'] = [r[0] for r in res]
        data['parsed'] = [','.join(r[1]) if r[1] else '' for r in res]
        data['hit'] = [int(r[2]) for r in res]
        data['lenient_parse'] = [r[3] for r in res]
        dump(data, get_intermediate_file_path(eval_file, '_score', 'xlsx'))

        by_cat = defaultdict(list)
        for cat, hit in zip(data['category'], data['hit']):
            by_cat[cat].append(hit)
        cat_acc = {k: sum(v) / len(v) for k, v in sorted(by_cat.items())}
        check = release_check(data)
        ret = {
            # Headline = official macro_type_accuracy (mean over the "<level> / <type>" categories).
            # The official scorer withholds it for partial runs; here it is reported over the
            # categories present and evaluation_scope says "subset".
            'Overall': 100 * sum(cat_acc.values()) / len(cat_acc),
            'micro_accuracy': 100 * sum(data['hit']) / len(data),
        }
        for lv in LEVEL_ORDER:
            accs = [a for k, a in cat_acc.items() if k.startswith(lv + ' / ')]
            if accs:
                ret[f'{lv}_macro'] = 100 * sum(accs) / len(accs)
                hits = [h for c, h in zip(data['category'], data['hit']) if c.startswith(lv + ' / ')]
                ret[f'{lv}_micro'] = 100 * sum(hits) / len(hits)
        for k, a in cat_acc.items():
            ret[k] = 100 * a
        ret['count'] = len(data)
        for s in ('invalid', 'error'):
            ret[f'n_{s}'] = int(sum(x == s for x in data['status']))
        # Diagnostic: answers the official parser accepts leniently (scored as upstream does).
        ret['n_lenient_parse'] = int(sum(bool(x) for x in data['lenient_parse']))
        ret['n_lenient_parse_correct'] = int(sum(bool(x) and bool(h)
                                                 for x, h in zip(data['lenient_parse'], data['hit'])))
        ret['evaluation_scope'] = 'full' if check['matches_release'] else 'subset'
        ret['mode'] = self.mode
        ret['protocol'] = f'{sc.PROTOCOL}@{OFFICIAL_CODE_COMMIT[:7]}'
        dump(ret, get_intermediate_file_path(eval_file, '_acc', 'json'))
        return ret
