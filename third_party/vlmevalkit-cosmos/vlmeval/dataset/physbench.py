"""PhysBench (ICLR 2025): Benchmarking and Enhancing VLMs for Physical World Understanding.

Project page: https://physbench.github.io/   Paper: arXiv 2501.16411
Official eval code: USC-GVL/PhysBench @ a7eadc4da554e20163303ec0c1e076a35a9f52e2 (eval/), Apache-2.0
Data: HF USC-PSI-Lab/PhysBench @ 478fd93da8ec8d6f5252b9586b1fa10f335c5a95 (apache-2.0; the README's
USC-GVL/PhysBench name redirects there). test.json holds the 10,002 questions (no labels); labels are
public in the official repo (eval/physbench/test_answer.json, the file print_test_score.py scores
against and which the README says gives the EvalAI numbers).

Keys
  * PhysBench      all 10,002 items (test 9,802 + val 200). The paper / leaderboard report the full
                   set including the val split (paper App. B.8; lmms-eval physbench.yaml).
  * PhysBench_val  the 200-item val split (split == 'val'), scored against val_answer.json.

Prompt (official eval/eval_utils/task_evaluator.py, `general` mode = interleaved models such as GPT-4o,
InternVL2.5, NVILA): `question + END_PROMPT`; the question's `<video>` / `<image>` placeholders are
replaced in order by the files of `file_name` (the official test.json guarantees the order; counts
match for all 10,002 items). Text pieces between placeholders are kept as separate text messages,
so the request is interleaved exactly like the official GPT-4o client (`_replace_placeholders`).
The GPT/Gemini clients additionally prefix a "The video is split to a series of images ..." line
when a video is present; that line describes frame lists and is not used here (Cosmos receives a
real video), as for the official open-source interleaved models.

Frames: official default `test_frame = 8` frames per video. VIDEO_LLM clients get the mp4 as a video
message that pins `nframes=8` (CosmosReason* then ignores every model-level sampling attribute, see
PROTOCOL.md §5); qwen-vl-utils samples linspace(0, n-1, 8). Image-frame clients get the same 8 frames
(linspace, rounded) as images in the video's place. 19 clips have only 4-7 decoded frames: they get all
frames rounded down to an even count (qwen-vl-utils needs an even nframes <= total; see frame_budget).

Scoring (headline, official bit-for-bit): `pred == gt or pred.startswith(gt)` on the raw prediction
string (text after the last `</think>` for *_think models); `Overall` = correct / number of scored
items; per task_type / sub_type / ability_type accuracy = correct / total of that type. This is the
verbatim official `calculate_accuracy` (utils/physbench_official.py) and is asserted equal to our
per-row computation on every evaluate(). Official label quirks are kept: 17 items have an empty
label (every prediction startswith '' -> always correct), 7 have non-letter / multi labels
('d', ' D', '1', 'E', 'A B C', ...).
Diagnostics (not the headline):
  * `lmms_overall` etc.: lmms-eval tasks/physbench @ 9134731 (first character in ABCD, exact match,
    the 17 empty-label rows dropped).
  * `Overall_lenient`: official rule, but when it fails and the kit's can_match_option extracts a
    letter from the reply, that letter is compared instead (e.g. "(B)", "The answer is B").
"""

import hashlib
import json
import os
import re
import urllib.request
import zipfile
from collections import OrderedDict

import numpy as np
import pandas as pd
import portalocker

from vlmeval.smp import LMUDataRoot, dump, load
from vlmeval.smp.file import get_intermediate_file_path
from .video_base import VideoBaseDataset

HF_REPO = 'USC-PSI-Lab/PhysBench'
HF_REVISION = '478fd93da8ec8d6f5252b9586b1fa10f335c5a95'
OFFICIAL_REPO = 'USC-GVL/PhysBench'
OFFICIAL_COMMIT = 'a7eadc4da554e20163303ec0c1e076a35a9f52e2'
SHA256 = {
    'test.json': 'd67c12a4d595ee9471ad079b90874e03dbbb8db3c5242ea3076c69affe876b8d',
    'image.zip': '629b7cb64b328a9f131c18b45c4038f5e0bdd945324db4fbd2f4309f44cf942d',
    'video.zip': '0bc9e11072e8ea8bc15c11ad62aa030f080684752e96e14803bdf4c963e378f3',
    'test_answer.json': '229ddf9be156dbb93f23af5fc714addb144a781638dff89a571034c907f12446',
    'val_answer.json': 'b960d24d8f573ddc1a8a4472e0684065b4f28b90a96c7f34947b5c8c45d15940',
}
BUILD_VERSION = 2  # 2: video_frames / video_nframes columns

# official eval/eval_utils/task_evaluator.py
TEST_FRAME = 8
END_PROMPT = ("\nAnswer with the option's letter from the given choices directly. "
              "You can only answer one letter from A, B, C, or D.")

SUBSETS = {'PhysBench': ('all', 'test_answer.json', 10002), 'PhysBench_val': ('val', 'val_answer.json', 200)}

TASK_ORDER = ['property', 'relationships', 'scene', 'dynamics']
SUB_ORDER = {
    'property': ['number', 'mass', 'color', 'attribute'],
    'relationships': ['size', 'location', 'depth', 'distance', 'motion'],
    'scene': ['temperature', 'viewpoint', 'air', 'light'],
    'dynamics': ['collision', 'throwing', 'manipulation', 'fluid', 'chemistry', 'others'],
}
ABILITY_ORDER = ['identify', 'comparison', 'static', 'dynamic', 'perception', 'prediction', 'judgment', 'reasoning']
PLACEHOLDER_RE = re.compile(r'(<video>|<image>)')


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------
def as_prediction(x) -> str:
    """Prediction cell -> string (NaN / None -> '')."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    return str(x)


def official_hit(pred: str, gt: str) -> bool:
    """eval/print_test_score.py answer_true()."""
    return pred == gt or pred.startswith(gt)


def lmms_letter(pred: str) -> str:
    """lmms-eval physbench_process_results(): first character if it is A-D (case-insensitive)."""
    pred = pred.strip() if pred else ''
    return pred[0].upper() if pred and pred[0].upper() in 'ABCD' else ''


def lenient_hit(pred: str, gt: str) -> bool:
    if official_hit(pred, gt):
        return True
    from .utils.spatial_bench.matching_func import can_match_option
    letter = can_match_option(pred, choices=['A', 'B', 'C', 'D'])
    return bool(letter) and official_hit(letter, gt)


def score_rows(preds, gts):
    """Per-row official / lenient / lmms results. preds, gts: aligned lists of str."""
    hit = [official_hit(p, g) for p, g in zip(preds, gts)]
    len_hit = [h or lenient_hit(p, g) for h, p, g in zip(hit, preds, gts)]
    lmms = [lmms_letter(p) for p in preds]
    lmms_hit = [bool(g) and lm == g.upper() for lm, g in zip(lmms, gts)]
    return hit, len_hit, lmms, lmms_hit


def summarize(df):
    """df columns: hit, hit_lenient, lmms_hit, has_label, task_type, sub_type, ability_type."""
    n = len(df)
    out = OrderedDict()
    out['Overall'] = 100.0 * df['hit'].sum() / n if n else 0.0
    for t in TASK_ORDER:
        g = df[df['task_type'] == t]
        if len(g):
            out[t] = 100.0 * g['hit'].sum() / len(g)
    for t in TASK_ORDER:
        for s in SUB_ORDER[t]:
            g = df[df['sub_type'] == s]
            if len(g):
                out[f'{t}/{s}'] = 100.0 * g['hit'].sum() / len(g)
    for a in ABILITY_ORDER:
        g = df[df['ability_type'] == a]
        if len(g):
            out[f'ability/{a}'] = 100.0 * g['hit'].sum() / len(g)
    out['n'] = n
    out['Overall_lenient'] = 100.0 * df['hit_lenient'].sum() / n if n else 0.0
    lab = df[df['has_label']]
    out['lmms_n'] = len(lab)
    out['lmms_overall'] = 100.0 * lab['lmms_hit'].sum() / len(lab) if len(lab) else 0.0
    for t in TASK_ORDER:
        g = lab[lab['task_type'] == t]
        if len(g):
            out[f'lmms_{t}'] = 100.0 * g['lmms_hit'].sum() / len(g)
    return out


# ----------------------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------------------
def physbench_root() -> str:
    return os.path.join(LMUDataRoot(), 'PhysBench')


def _sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(bufsize)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _fetch(name, dst):
    """Download one pinned file (HF data or official-repo label file) and check its sha256."""
    if name in ('test_answer.json', 'val_answer.json'):
        url = (f'https://raw.githubusercontent.com/{OFFICIAL_REPO}/{OFFICIAL_COMMIT}/'
               f'eval/physbench/{name}')
        tmp = f'{dst}.{os.getpid()}.part'
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
    else:
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_REPO, name, repo_type='dataset', revision=HF_REVISION,
                        local_dir=os.path.dirname(dst))


def _ensure_file(root, name):
    path = os.path.join(root, 'hf', name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(2):
        if not os.path.exists(path):
            _fetch(name, path)
        digest = _sha256(path)
        if digest == SHA256[name]:
            return path
        os.remove(path)
    raise RuntimeError(f'{path}: sha256 {digest} != pinned {SHA256[name]}')


def _missing(root, files):
    out = []
    for f in files:
        p = media_path(root, f)
        if not (os.path.isfile(p) and os.path.getsize(p) > 0):
            out.append(f)
    return out


def media_path(root, fname):
    sub = 'video' if fname.lower().endswith('.mp4') else 'image'
    return os.path.join(root, sub, fname)


def _extract(zip_path, dst_dir, wanted):
    """Extract the members whose basename is in `wanted` into dst_dir/<basename> (flat, path-safe)."""
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = os.path.basename(info.filename)
            if base not in wanted:
                continue
            out = os.path.join(dst_dir, base)
            tmp = f'{out}.{os.getpid()}.part'
            with zf.open(info) as src, open(tmp, 'wb') as dst:
                while True:
                    buf = src.read(1 << 22)
                    if not buf:
                        break
                    dst.write(buf)
            os.replace(tmp, out)
            n += 1
    return n


def frame_budget(n_frames: int) -> int:
    """Frames sent for a video with n_frames decoded frames: the official 8, or - for the 19 clips
    with fewer than 8 frames (4-7) - all of them rounded down to an even count, because qwen-vl-utils
    requires 2 <= nframes <= total and a multiple of 2 (the official clients send all frames of such
    clips: GPT stride 1, opencv path pads with black frames)."""
    return min(TEST_FRAME, n_frames - n_frames % 2)


def video_frame_counts(root, videos):
    """Decoded frame count per mp4 (decord, the reader qwen-vl-utils uses), cached in
    <root>/video_frames.json keyed by name with the file size."""
    import decord
    cache_path = os.path.join(root, 'video_frames.json')
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    out, dirty = {}, False
    for v in videos:
        p = media_path(root, v)
        size = os.path.getsize(p)
        hit = cache.get(v)
        if not hit or hit[1] != size:
            hit = [len(decord.VideoReader(p)), size]
            cache[v] = hit
            dirty = True
        out[v] = hit[0]
    if dirty:
        tmp = f'{cache_path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(cache, f)
        os.replace(tmp, cache_path)
    return out


def load_items(root):
    """test.json joined with the official labels (test_answer.json covers all 10,002 idx)."""
    with open(_ensure_file(root, 'test.json'), encoding='utf-8') as f:
        items = json.load(f)
    with open(_ensure_file(root, 'test_answer.json'), encoding='utf-8') as f:
        labels = {x['idx']: x for x in json.load(f)}
    assert len(items) == 10002 and len(labels) == 10002, 'unexpected PhysBench size'
    rows = []
    for it in items:
        lab = labels[it['idx']]
        files = list(it['file_name'])
        q = it['question']
        n_vid = sum(f.lower().endswith('.mp4') for f in files)
        assert q.count('<video>') == n_vid and q.count('<image>') == len(files) - n_vid, it['idx']
        rows.append(dict(
            index=int(it['idx']), idx=int(it['idx']), split=it['split'], mode=it['mode'],
            source=it.get('source'), question=q, file_name=json.dumps(files),
            video=next((f for f in files if f.lower().endswith('.mp4')), 'none'),
            answer=lab['answer'], task_type=lab['task_type'], sub_type=lab['sub_type'],
            ability_type=lab['ability_type']))
    return rows


class PhysBench(VideoBaseDataset):
    # 'MCQ' keeps CosmosReason*.use_custom_prompt() False, so build_prompt() below is used.
    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='PhysBench', pack=False, nframe=0, fps=-1, **kwargs):
        if dataset not in SUBSETS:
            raise ValueError(f'Unknown PhysBench key {dataset}; choose from {list(SUBSETS)}')
        if (nframe or 0) > 0 and int(nframe) != TEST_FRAME:
            raise ValueError(f'{dataset}: official frame budget is {TEST_FRAME}, got nframe={nframe}')
        if fps is not None and fps > 0:
            raise ValueError(f'{dataset}: fps sampling is not supported (official {TEST_FRAME} frames)')
        self.split, self.label_file, self.expected = SUBSETS[dataset]
        super().__init__(dataset=dataset, pack=pack, nframe=TEST_FRAME, fps=-1)

    @classmethod
    def supported_datasets(cls):
        return list(SUBSETS)

    def prepare_dataset(self, dataset):
        root = physbench_root()
        os.makedirs(root, exist_ok=True)
        tsv = os.path.join(LMUDataRoot(), f'{dataset}.tsv')
        manifest = os.path.join(LMUDataRoot(), f'{dataset}.manifest.json')
        with portalocker.Lock(os.path.join(root, '.prepare.lock'), 'w', timeout=7200):
            rows = load_items(root)
            if self.split != 'all':
                rows = [r for r in rows if r['split'] == self.split]
            assert len(rows) == self.expected, (dataset, len(rows))
            files = sorted({f for r in rows for f in json.loads(r['file_name'])})
            missing = _missing(root, files)
            if missing:
                for z, sub in (('image.zip', 'image'), ('video.zip', 'video')):
                    want = {os.path.basename(f) for f in missing if media_path(root, f).startswith(
                        os.path.join(root, sub) + os.sep)}
                    if want:
                        _extract(_ensure_file(root, z), os.path.join(root, sub), want)
                missing = _missing(root, files)
                if missing:
                    raise RuntimeError(f'{len(missing)} PhysBench media files missing after extraction, '
                                       f'e.g. {missing[:3]}')
            counts = video_frame_counts(root, [f for f in files if f.lower().endswith('.mp4')])
            for r in rows:
                n = counts.get(r['video'])
                r['video_frames'] = n if n is not None else 0
                r['video_nframes'] = frame_budget(n) if n is not None else 0
            if self.split != 'all':
                # val labels: the official val_answer.json has the same ids and answers as the
                # test_answer.json rows (14 rows carry an older task/ability taxonomy there; the
                # official val scorer uses val_answer.json, and so does evaluate()).
                with open(_ensure_file(root, self.label_file), encoding='utf-8') as f:
                    val = {x['idx']: x for x in json.load(f)}
                assert set(val) == {r['idx'] for r in rows}
                for r in rows:
                    assert val[r['idx']]['answer'] == r['answer'], r['idx']
            want_manifest = dict(build_version=BUILD_VERSION, hf_repo=HF_REPO, hf_revision=HF_REVISION,
                                 official_commit=OFFICIAL_COMMIT, n=len(rows),
                                 test_json_sha256=SHA256['test.json'],
                                 labels_sha256=SHA256['test_answer.json'])
            current = None
            if os.path.exists(manifest) and os.path.exists(tsv):
                with open(manifest) as f:
                    current = json.load(f)
                if current.get('tsv_md5') != hashlib.md5(open(tsv, 'rb').read()).hexdigest():
                    current = None
            if current is None or any(current.get(k) != v for k, v in want_manifest.items()):
                tmp = f'{tsv}.{os.getpid()}.tmp.tsv'
                pd.DataFrame(rows).to_csv(tmp, sep='\t', index=False)
                os.replace(tmp, tsv)
                want_manifest['tsv_md5'] = hashlib.md5(open(tsv, 'rb').read()).hexdigest()
                with open(manifest + '.tmp', 'w') as f:
                    json.dump(want_manifest, f, indent=1)
                os.replace(manifest + '.tmp', manifest)
        return dict(root=root, data_file=tsv)

    # ------------------------------------------------------------------------------------------
    def save_video_frames(self, video):
        """The official 8-frame budget as jpgs (image-frame clients): linspace(0, n-1, 8), rounded."""
        import decord
        from PIL import Image

        stem = os.path.splitext(video)[0]
        frame_paths = self.frame_paths(stem)
        if all(os.path.exists(p) for p in frame_paths):
            return frame_paths
        lock_path = os.path.join(self.frame_root, stem + '.lock')
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with portalocker.Lock(lock_path, 'w', timeout=300):
            if all(os.path.exists(p) for p in frame_paths):
                return frame_paths
            vid = decord.VideoReader(media_path(self.data_root, video))
            idx = np.linspace(0, len(vid) - 1, TEST_FRAME).round().astype(int)
            for i, pth in zip(idx, frame_paths):
                if not os.path.exists(pth):
                    tmp = pth + '.tmp.jpg'
                    Image.fromarray(vid[int(i)].asnumpy()).save(tmp)
                    os.replace(tmp, pth)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        files = json.loads(line['file_name'])
        prompt = line['question'] + END_PROMPT
        message, k = [], 0
        for part in PLACEHOLDER_RE.split(prompt):
            if part in ('<video>', '<image>'):
                f = files[k]
                k += 1
                assert f.lower().endswith('.mp4') == (part == '<video>'), (line['index'], f, part)
                if part == '<image>':
                    message.append(dict(type='image', value=media_path(self.data_root, f)))
                elif video_llm:
                    # nframes pinned to the official budget (overrides model-level sampling); < 8 only
                    # for the 19 clips shorter than 8 frames (frame_budget). One video per item at most.
                    message.append(dict(type='video', value=media_path(self.data_root, f),
                                        nframes=int(line["video_nframes"])))
                else:
                    message.extend(dict(type='image', value=p) for p in self.save_video_frames(f))
            elif part:
                message.append(dict(type='text', value=part))
        assert k == len(files), line['index']
        return message

    # ------------------------------------------------------------------------------------------
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.physbench_official import calculate_accuracy

        data = load(eval_file)
        root = physbench_root()
        with open(_ensure_file(root, self.label_file), encoding='utf-8') as f:
            labels = {x['idx']: x for x in json.load(f)}
        data = data.sort_values('index').reset_index(drop=True)
        preds = [as_prediction(p) for p in data['prediction']]
        gts = [labels[int(i)]['answer'] for i in data['index']]
        hit, len_hit, lmms, lmms_hit = score_rows(preds, gts)
        df = pd.DataFrame(dict(
            index=data['index'].astype(int), prediction=preds, answer=gts,
            task_type=[labels[int(i)]['task_type'] for i in data['index']],
            sub_type=[labels[int(i)]['sub_type'] for i in data['index']],
            ability_type=[labels[int(i)]['ability_type'] for i in data['index']],
            hit=hit, hit_lenient=len_hit, lmms_pred=lmms, lmms_hit=lmms_hit,
            has_label=[bool(g) for g in gts]))
        summary = summarize(df)

        # Cross-check with the verbatim official scorer on the same predictions.
        sub_file = get_intermediate_file_path(eval_file, '_physbench_submission', 'json')
        with open(sub_file, 'w', encoding='utf-8') as f:
            json.dump([{'idx': int(i), 'answer': p} for i, p in zip(df['index'], preds)], f,
                      ensure_ascii=False, indent=1)
        label_path = os.path.join(root, 'hf', self.label_file)
        off = calculate_accuracy(label_path, sub_file)
        assert off['overall_correct'] == int(df['hit'].sum()) and off['overall_total'] == len(df)
        for t, v in off['task_type_accuracy'].items():
            assert abs(v['accuracy'] - summary[t]) < 1e-9, (t, v, summary[t])
        summary['official_overall'] = off['overall_accuracy']
        summary['evaluation_scope'] = 'full' if (len(df) == self.expected and df['index'].is_unique) else 'subset'

        score_file = get_intermediate_file_path(eval_file, '_physbench_score', 'xlsx')
        dump(df, score_file)
        table = pd.DataFrame([dict(key=k, value=v) for k, v in summary.items()])
        table_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        table.to_csv(table_file, index=False)
        print(f'[{self.dataset_name}] ({table_file}):\n' + table.to_string(index=False))
        return summary
