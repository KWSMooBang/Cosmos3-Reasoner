"""SpaceCast-Bench: predictive spatial reasoning over multi-view indoor images.

Reference:
  SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models
  https://arxiv.org/abs/2610.12402
  data: https://huggingface.co/datasets/hongxingli/SpaceCast-Bench (pinned revision below)
  code: https://github.com/ZJU-REAL/SpaceCast-Bench (protocol "spacecast-v1")

Local addition (Cosmos3-Reasoner eval repo; not in NVIDIA upstream).

Two keys, one per official evaluation mode:
  SpaceCastBench      -> official `--mode direct` (answer letter only)
  SpaceCastBench_CoT  -> official `--mode cot`    (brief reasoning, last line "Answer: <letter>")

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
"""
import ast
import hashlib
import json
import os
import os.path as osp
import warnings
from collections import defaultdict

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset
from .utils import spacecast_official as sc

HF_REPO = 'hongxingli/SpaceCast-Bench'
HF_REVISION = '45be662ca54fd6f89c25b1443a54825fb03d976e'   # 2026-10-16
HF_FILE = 'data/test-00000-of-00001.parquet'
OFFICIAL_CODE_COMMIT = '6696e5eb41371a9c8038c0d902d2aa488b800593'
FAIL_MSG = 'Failed to obtain answer via API.'
TSV_NAME = 'SpaceCastBench.tsv'   # shared by both modes
IMG_DIR = 'SpaceCastBench'
LEVEL_ORDER = ('L1', 'L2', 'L3')


def _official_row(line):
    """The fields the official build_prompt / parse_response read, from one TSV row."""
    return {'question': line['question'], 'options': json.loads(line['options']),
            'multi_select': _as_bool(line['multi_select'])}


def _as_bool(v):
    if isinstance(v, str):
        return v.strip().lower() == 'true'
    return bool(v)


def build_tsv(parquet_path, tsv_path, img_root):
    """Convert the HF parquet to a meta-only TSV plus preprocessed JPEGs under img_root.

    Images are stored once per `image_paths` entry (dataset/scene/frame); a path that maps to two
    different preprocessed images raises.
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
                    with open(out + '.tmp', 'wb') as f:
                        f.write(blob)
                    os.replace(out + '.tmp', out)
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
    tmp = tsv_path + '.tmp'
    df.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, tsv_path)
    return df


def release_check(df):
    counts = df['category'].value_counts().to_dict()
    scenes = {(d, s) for d, s in zip(df['dataset'], df['scene_id'])}
    return {'questions': len(df), 'scenes': len(scenes), 'categories': len(counts),
            'matches_release': (len(df) == sc.RELEASE_QUESTIONS and len(scenes) == sc.RELEASE_SCENES
                                and counts == sc.RELEASE_CATEGORY_COUNTS)}


def score_rows(rows):
    """Per-question correctness exactly as spacecast_eval.scoring.score.

    rows: iterable of dicts with question/options (list)/multi_select/answer/prediction, where
    prediction is the raw model text (None / NaN / FAIL_MSG = no answer).
    Returns a list of (status, parsed letters or None, correct) tuples.
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
            out.append(('error', None, False))
            continue
        parsed = sc.parse_prediction(sc.parse_response(raw, row), allowed, multi)
        status = 'valid' if parsed is not None else 'invalid'
        out.append((status, sorted(parsed) if parsed is not None else None, parsed == target))
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
        if not osp.exists(tsv_path):
            parquet = osp.join(root, 'SpaceCast-Bench', 'hf', osp.basename(HF_FILE))
            if not osp.exists(parquet):
                from huggingface_hub import hf_hub_download
                parquet = hf_hub_download(HF_REPO, HF_FILE, repo_type='dataset', revision=HF_REVISION)
            build_tsv(parquet, tsv_path, osp.join(root, 'images', IMG_DIR))
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
        data['status'] = [s for s, _, _ in res]
        data['parsed'] = [','.join(p) if p else '' for _, p, _ in res]
        data['hit'] = [int(c) for _, _, c in res]
        dump(data, get_intermediate_file_path(eval_file, '_score', 'xlsx'))

        by_cat = defaultdict(list)
        for cat, hit in zip(data['category'], data['hit']):
            by_cat[cat].append(hit)
        cat_acc = {k: sum(v) / len(v) for k, v in sorted(by_cat.items())}
        full = len(data) == sc.RELEASE_QUESTIONS and len(cat_acc) == len(sc.RELEASE_CATEGORY_COUNTS)
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
        ret['evaluation_scope'] = 'full' if full else 'subset'
        ret['mode'] = self.mode
        ret['protocol'] = f'{sc.PROTOCOL}@{OFFICIAL_CODE_COMMIT[:7]}'
        dump(ret, get_intermediate_file_path(eval_file, '_acc', 'json'))
        return ret
