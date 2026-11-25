"""VSTI-Bench (VLM-3R, CVPR 2026): Visual-Spatial-Temporal Intelligence benchmark.

Project page: https://vlm-3r.github.io/   Paper: arXiv 2505.20279 (v5)   Data: HF Journey9ni/vstibench

Ported from the official evaluation code (prompt, question types, answer parsing, MRA, aggregation):
  * VITA-Group/VLM-3R  thinking-in-space/lmms_eval/tasks/vstibench/{vstibench.yaml,utils.py}
    @ 206afd0ece4369457fb3c5c7781a6735fd2bbe8c
  * frames: VITA-Group/VLM-3R scripts/VSTI-Bench/eval_vlm_3r_vstibench.sh (max_frames_num=32) +
    lmms_eval/models/vlm_3r.py load_video (np.linspace(0, n-1, 32, dtype=int) over the mp4).
Data pinned to HF revision 0903eb0b0c9d03da730d43503c48302a1f3b412d (post-erratum labels, 2026-07-14:
5,736 questions; the paper's Table 2 used the pre-erratum 6,042-question release @ 27ff0a2, which the
authors say is not comparable). The cache ($LMUData/VSTIBench/, VSTIBench.tsv) is validated on every load
against $LMUData/VSTIBench.manifest.json (revision, pinned file hashes, TSV md5, size + sha256 of every
mp4) and rebuilt on mismatch (validate_cache / build_cache). evaluate() reports evaluation_scope
('full' only for the exact release) and headline_partial; Overall on a subset is not the release metric.

Frames. VSTI-Bench videos are the full-length ScanNet val mp4s (640x480, 24 fps). The official eval
samples 32 frames uniformly. The question texts refer to "frame k of 32".
  * VIDEO_LLM clients (CosmosReason*): the mp4 is sent as a video message carrying `nframes=32`
    (pinned like ReVSI: CosmosReason._get_video_processing_kwargs then ignores all model-level
    sampling attributes). qwen-vl-utils samples torch.linspace(0, n-1, 32).round(), the official
    loader np.linspace(0, n-1, 32, dtype=int) (floor): indices differ by at most one source
    frame (1/24 s); scripts/check_vstibench_frames.py measures this on the real files.
  * image-frame clients: exactly the official indices np.linspace(0, n-1, 32, dtype=int).

Scoring (official, bit-for-bit, see score_dataframe):
  * MCA: fuzzy_matching(pred) = pred.split(' ')[0].rstrip('.').strip(), case-insensitive == mc_answer.
  * NA: float(fuzzy_matching(pred)) -> MRA over thresholds linspace(.5, .95, 10), `<=`; unparsable -> 0.
  Headline `Overall` = paper Table 2 "Avg" = mean of the 5 task groups, where a group is the mean of
  its sub-type means (Cam-Obj Rel. Dist = v1,v2,v3; Obj-Obj Rel. Pos = lr,nf,ud). Evidence: the
  paper's chance rows (Rel. Dist 36.1 = mean(1/4,1/3,1/2), not the pooled 42.8) and VLM-3R / Gemini
  Avg = mean of the 5 columns. `Overall_code9` = what the official lmms-eval aggregator returns
  (unweighted mean of the 9 per-question-type scores; it does not merge sub-types).
  Diagnostic `*_lenient` (as ReVSI): official parse, falling back to the kit's rule extractors only
  when the official token is not a valid answer.
"""

import json
import math
import os
import re
import tarfile
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import portalocker

from vlmeval.smp import LMUDataRoot, dump, load
from vlmeval.smp.file import get_intermediate_file_path
from .video_base import VideoBaseDataset

HF_REPO = 'Journey9ni/vstibench'
HF_REVISION = '0903eb0b0c9d03da730d43503c48302a1f3b412d'
# @ HF_REVISION: ScanNet.tar.gz sha256 = its LFS oid; test.json (not LFS) sha256 of the file whose git blob id
# equals the Hub's (bc83433ba242cfb7a6651f237f61608b3193ce38).
TAR_SHA256 = '33339aa0348096558a95e780c997c84baef8e50d1f40b62f31f976525ab9f3cb'
TEST_JSON_SHA256 = '7c0780c0f03355446c958123f013a11773130d83bf12001936210bafffd0ae94'
BUILD_VERSION = 1
NUM_FRAMES = 32

# --- verbatim from vstibench.yaml (lmms_eval_specific_kwargs.default) + utils.vsibench_doc_to_text ---
PRE_PROMPT = 'These are frames of a video.'   # yaml pre_prompt "" -> utils fallback
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NA_POST_PROMPT = 'Please answer the question using a single word or phrase.'

MCA_QUESTION_TYPES = [
    'obj_obj_relative_pos_nf',
    'obj_obj_relative_pos_ud',
    'obj_obj_relative_pos_lr',
    'camera_obj_rel_dist_v1',
    'camera_obj_rel_dist_v2',
    'camera_obj_rel_dist_v3',
    'camera_movement_direction',
]
NA_QUESTION_TYPES = [
    'camera_obj_abs_dist',
    'camera_displacement',
    'camera_obj_dist_change',   # listed upstream; no question of this type in the release
]

# Paper Table 2 column order: Cam-Obj Abs. Dist, Cam. Displace, Cam. Mov. Dir, Obj-Obj Rel. Pos,
# Cam-Obj Rel. Dist.
TASKS = OrderedDict([
    ('camera_obj_abs_dist', ['camera_obj_abs_dist']),
    ('camera_displacement', ['camera_displacement']),
    ('camera_movement_direction', ['camera_movement_direction']),
    ('obj_obj_relative_pos', ['obj_obj_relative_pos_lr', 'obj_obj_relative_pos_nf', 'obj_obj_relative_pos_ud']),
    ('camera_obj_rel_dist', ['camera_obj_rel_dist_v1', 'camera_obj_rel_dist_v2', 'camera_obj_rel_dist_v3']),
])
TASK_LABELS = {
    'camera_obj_abs_dist': 'Cam-Obj Abs. Dist', 'camera_displacement': 'Cam. Displace',
    'camera_movement_direction': 'Cam. Mov. Dir', 'obj_obj_relative_pos': 'Obj-Obj Rel. Pos',
    'camera_obj_rel_dist': 'Cam-Obj Rel. Dist',
}
# Release counts @ HF_REVISION (checked when the TSV is built).
RELEASE_COUNTS = {
    'camera_displacement': 833, 'camera_obj_abs_dist': 905, 'camera_movement_direction': 913,
    'camera_obj_rel_dist_v1': 91, 'camera_obj_rel_dist_v2': 493, 'camera_obj_rel_dist_v3': 856,
    'obj_obj_relative_pos_lr': 605, 'obj_obj_relative_pos_nf': 556, 'obj_obj_relative_pos_ud': 484,
}


# ----------------------------------------------------------------------------------------------
# Official scoring, verbatim (utils.py @ 206afd0)
# ----------------------------------------------------------------------------------------------
def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()


def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred


def official_score(question_type, prediction, mc_answer, ground_truth) -> float:
    """utils.vsibench_process_results for one doc (accuracy / MRA:.5:.95:.05)."""
    prediction = str(prediction)
    if question_type in MCA_QUESTION_TYPES:
        return float(exact_match(fuzzy_matching(prediction), str(mc_answer)))
    if question_type in NA_QUESTION_TYPES:
        try:
            return float(mean_relative_accuracy(to_float(fuzzy_matching(prediction)), to_float(ground_truth)))
        except TypeError:
            return 0.
    raise ValueError(f'Unknown question type: {question_type}')


def official_aggregate(question_types, scores) -> float:
    """utils.vsibench_aggregate_results: mean over per-question-type means, x100."""
    df = pd.DataFrame({'question_type': list(question_types), 'score': list(scores)})
    per = df.groupby('question_type')['score'].mean()
    return float(per.sum() / len(per)) * 100.


# ----------------------------------------------------------------------------------------------
# Lenient diagnostic (not official)
# ----------------------------------------------------------------------------------------------
_THOUSANDS_RE = re.compile(r'(?<=\d),(?=\d{3}(?!\d))')


def _option_letters(options) -> str:
    opts = parse_options(options)
    letters = ''.join(str(o).strip()[:1].upper() for o in opts if str(o).strip())
    return letters or 'ABCD'


def lenient_parse(question_type, prediction, options=None):
    """(parsed answer or None, mode in {'official','fallback','fail'})."""
    from .utils.spatial_bench.matching_func import can_match_na, can_match_option

    text = '' if prediction is None or (isinstance(prediction, float) and math.isnan(prediction)) \
        else str(prediction)
    tok = fuzzy_matching(text)
    if question_type in MCA_QUESTION_TYPES:
        letters = _option_letters(options)
        if len(tok) == 1 and tok.upper() in letters:
            return tok.upper(), 'official'
        letter = can_match_option(text, choices=list(letters)) if text.strip() else False
        if letter and letter in letters:
            return letter, 'fallback'
        return None, 'fail'
    if question_type in NA_QUESTION_TYPES:
        v = to_float(tok)
        if v is not None and math.isfinite(v):
            return v, 'official'
        v = can_match_na(_THOUSANDS_RE.sub('', text)) if text.strip() else None
        if v is not None and math.isfinite(v):
            return v, 'fallback'
        return None, 'fail'
    raise ValueError(f'Unknown question type: {question_type}')


def lenient_score(question_type, prediction, mc_answer, ground_truth, options=None):
    parsed, mode = lenient_parse(question_type, prediction, options)
    if parsed is None:
        return 0.0, parsed, mode
    if question_type in MCA_QUESTION_TYPES:
        return float(str(parsed).lower() == str(mc_answer).strip().lower()), parsed, mode
    try:
        return float(mean_relative_accuracy(float(parsed), float(ground_truth))), parsed, mode
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0, parsed, mode


# ----------------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------------
def _groups(df: pd.DataFrame, col: str) -> OrderedDict:
    per_qtype = df.groupby('question_type')[col].mean().to_dict()
    out = OrderedDict()
    for task, qtypes in TASKS.items():
        vals = [per_qtype[q] for q in qtypes if q in per_qtype]
        if vals:
            out[task] = float(np.mean(vals))
    return out


def score_dataframe(df: pd.DataFrame):
    """Score a prediction table (question_type, prediction, mc_answer, ground_truth, options).

    Returns (per-row DataFrame, summary dict 0-100, per-task table DataFrame).
    """
    df = df.copy()
    if 'options' not in df.columns:
        df['options'] = None
    off, len_, parsed, modes = [], [], [], []
    for _, r in df.iterrows():
        pred = r['prediction']
        pred = '' if pred is None or (isinstance(pred, float) and math.isnan(pred)) else pred
        off.append(official_score(r['question_type'], pred, r['mc_answer'], r['ground_truth']))
        s, p, m = lenient_score(r['question_type'], pred, r['mc_answer'], r['ground_truth'], r['options'])
        len_.append(s)
        parsed.append(p)
        modes.append(m)
    df['score'] = off
    df['score_lenient'] = len_
    df['pred_parsed_lenient'] = parsed
    df['parse_mode'] = modes

    g_off, g_len = _groups(df, 'score'), _groups(df, 'score_lenient')
    summary = OrderedDict()
    summary['Overall'] = 100.0 * float(np.mean(list(g_off.values()))) if g_off else 0.0
    summary['Overall_lenient'] = 100.0 * float(np.mean(list(g_len.values()))) if g_len else 0.0
    summary['Overall_code9'] = official_aggregate(df['question_type'], df['score']) if len(df) else 0.0
    for t in g_off:
        summary[t] = 100.0 * g_off[t]
    for q, v in df.groupby('question_type')['score'].mean().items():
        summary[f'{q}_{"accuracy" if q in MCA_QUESTION_TYPES else "MRA:.5:.95:.05"}'] = 100.0 * float(v)
    for t in g_len:
        summary[f'{t}_lenient'] = 100.0 * g_len[t]

    rows = []
    task_of = {q: t for t, qs in TASKS.items() for q in qs}
    df['task'] = df['question_type'].map(task_of)
    for t in g_off:
        sub = df[df['task'] == t]
        rows.append(dict(
            task=TASK_LABELS[t], key=t, metric='MRA' if TASKS[t][0] in NA_QUESTION_TYPES else 'Acc',
            n=len(sub), score=100.0 * g_off[t], score_lenient=100.0 * g_len[t],
            n_fallback=int((sub['parse_mode'] == 'fallback').sum()),
            n_unparsed=int((sub['parse_mode'] == 'fail').sum())))
    rows.append(dict(
        task='Avg', key='overall', metric='mean of 5 tasks', n=len(df), score=summary['Overall'],
        score_lenient=summary['Overall_lenient'], n_fallback=int((df['parse_mode'] == 'fallback').sum()),
        n_unparsed=int((df['parse_mode'] == 'fail').sum())))
    rows.append(dict(task='Avg (official code, 9 types)', key='overall_code9', metric='mean of 9 types',
                     n=len(df), score=summary['Overall_code9']))
    return df, summary, pd.DataFrame(rows)


def parse_options(options):
    if options is None or (isinstance(options, float) and math.isnan(options)):
        return []
    if isinstance(options, np.ndarray):
        return [str(o) for o in options.tolist()]
    if isinstance(options, (list, tuple)):
        return [str(o) for o in options]
    if isinstance(options, str):
        s = options.strip()
        if not s:
            return []
        return [str(o) for o in json.loads(s)]
    return [str(options)]


def official_frame_indices(n_frames, num=NUM_FRAMES):
    """vlm_3r.load_video: np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)."""
    return np.linspace(0, n_frames - 1, num, dtype=int).tolist()


# ----------------------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------------------
def vstibench_root() -> str:
    return os.path.join(LMUDataRoot(), 'VSTIBench')


def _sha256(path, bufsize=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(bufsize)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _download(root, patterns):
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type='dataset', revision=HF_REVISION, local_dir=root,
                      allow_patterns=patterns)


def _missing_videos(root, videos):
    return [v for v in videos
            if not (os.path.isfile(os.path.join(root, v)) and os.path.getsize(os.path.join(root, v)) > 0)]


def _extract_videos(tar_path, root, needed):
    """Extract the needed mp4 members of ScanNet.tar.gz into root/ (path-safe, regular files only)."""
    needed = set(needed)
    absroot = os.path.abspath(root)
    n = 0
    with tarfile.open(tar_path, 'r:gz') as tf:
        for m in tf:
            name = os.path.normpath(m.name).replace('\\', '/').lstrip('./')
            if not m.isfile() or name not in needed:
                continue
            dst = os.path.abspath(os.path.join(root, name))
            if not dst.startswith(absroot + os.sep):
                raise RuntimeError(f'Unsafe path in tar: {m.name}')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = tf.extractfile(m)
            with open(dst + '.tmp', 'wb') as out:
                while True:
                    buf = src.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)
            os.replace(dst + '.tmp', dst)
            n += 1
    return n


def _md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _ensure_file(root, name, sha256):
    """root/name present with the pinned sha256; otherwise delete and re-download once."""
    path = os.path.join(root, name)
    for _ in range(2):
        if not os.path.exists(path):
            _download(root, [name])
        if _sha256(path) == sha256:
            return path
        os.remove(path)
        meta = os.path.join(root, '.cache', 'huggingface', 'download', name + '.metadata')
        if os.path.exists(meta):
            os.remove(meta)
    raise RuntimeError(f'{path}: sha256 != pinned {sha256} @ {HF_REVISION} after re-download')


def _build_tsv_frame(docs):
    df = pd.DataFrame(docs)
    counts = df['question_type'].value_counts().to_dict()
    if counts != RELEASE_COUNTS:
        raise RuntimeError(f'test.json counts {counts} != release counts @ {HF_REVISION}')
    if df['id'].duplicated().any():
        raise RuntimeError('duplicate ids in test.json')
    return pd.DataFrame({
        'index': df['id'].astype(int),
        'id': df['id'].astype(int),
        'question_type': df['question_type'],
        'question': df['question'],
        'options': [json.dumps(o) if isinstance(o, list) else '' for o in df['options']],
        'mc_answer': [a if isinstance(a, str) else '' for a in df['mc_answer']],
        'ground_truth': df['ground_truth'].astype(str),
        # `answer` = what the question is scored against (letter for MCA, number for NA)
        'answer': [a if q in MCA_QUESTION_TYPES else g for q, a, g in
                   zip(df['question_type'], df['mc_answer'], df['ground_truth'].astype(str))],
        'video': df['video_path'],
    })


def n_zero_targets(question_types, ground_truths) -> int:
    """Numeric questions with target 0 (the official MRA would raise ZeroDivisionError; 0 in the release)."""
    return int(sum(1 for q, g in zip(question_types, ground_truths)
                   if q in NA_QUESTION_TYPES and to_float(g) == 0))


def build_cache(root, tsv, manifest):
    """Pinned test.json + ScanNet.tar.gz -> extracted mp4s, TSV and manifest (atomic writes)."""
    if os.path.exists(manifest):
        os.remove(manifest)
    test_json = _ensure_file(root, 'test.json', TEST_JSON_SHA256)
    docs = json.load(open(test_json))
    videos = sorted({d['video_path'] for d in docs})
    tar_path = _ensure_file(root, 'ScanNet.tar.gz', TAR_SHA256)
    n = _extract_videos(tar_path, root, videos)
    missing = _missing_videos(root, videos)
    if missing:
        raise RuntimeError(f'{len(missing)} videos missing after extraction ({n} extracted), e.g. {missing[:3]}')
    out = _build_tsv_frame(docs)
    tmp = f'{tsv}.{os.getpid()}.tmp.tsv'
    out.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, tsv)
    man = dict(
        hf_repo=HF_REPO, hf_revision=HF_REVISION, build_version=BUILD_VERSION,
        test_json_sha256=TEST_JSON_SHA256, tar_sha256=TAR_SHA256, tsv_md5=_md5(tsv),
        n_questions=len(out), n_zero_target=n_zero_targets(out['question_type'], out['ground_truth']),
        videos={v: [os.path.getsize(os.path.join(root, v)), _sha256(os.path.join(root, v))] for v in videos})
    tmp = f'{manifest}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, manifest)


def validate_cache(root, tsv, manifest):
    """None if the cache is the pinned release, else a short reason.

    Checks: manifest pins (repo, revision, build version, file hashes), test.json sha256, TSV md5, the
    video set equals the one test.json references, every video's size and sha256.
    """
    if not os.path.exists(manifest):
        return 'no manifest'
    try:
        man = json.load(open(manifest))
    except Exception as e:  # noqa: BLE001
        return f'unreadable manifest ({e})'
    pins = dict(hf_repo=HF_REPO, hf_revision=HF_REVISION, build_version=BUILD_VERSION,
                test_json_sha256=TEST_JSON_SHA256, tar_sha256=TAR_SHA256)
    for k, v in pins.items():
        if man.get(k) != v:
            return f'manifest {k}={man.get(k)!r} != {v!r}'
    test_json = os.path.join(root, 'test.json')
    if not os.path.exists(test_json) or _sha256(test_json) != TEST_JSON_SHA256:
        return 'test.json missing or sha256 mismatch'
    if not os.path.exists(tsv) or _md5(tsv) != man.get('tsv_md5'):
        return 'TSV missing or md5 mismatch'
    videos = sorted({d['video_path'] for d in json.load(open(test_json))})
    if sorted(man.get('videos', {})) != videos:
        return 'manifest video list != videos referenced by test.json'
    for v in videos:
        p = os.path.join(root, v)
        size, sha = man['videos'][v]
        if not os.path.isfile(p) or os.path.getsize(p) != size:
            return f'{v}: missing or size mismatch'
        if _sha256(p) != sha:
            return f'{v}: sha256 mismatch'
    return None


def evaluation_scope(indices, question_types, release_indices) -> dict:
    """'full' iff the scored rows are exactly the release (5,736 unique ids, all 9 types at release counts)."""
    idx = list(indices)
    counts = pd.Series(list(question_types)).value_counts().to_dict()
    n_dup = len(idx) - len(set(idx))
    full = (n_dup == 0 and set(idx) == set(release_indices) and len(idx) == sum(RELEASE_COUNTS.values())
            and counts == RELEASE_COUNTS)
    return dict(evaluation_scope='full' if full else 'subset', headline_partial=int(not full),
                n_scored=len(idx), n_duplicate_rows=n_dup, n_types=len(counts))


class VSTIBench(VideoBaseDataset):
    # 'MCQ' (official lmms-eval: generate_until): keeps CosmosReason*.use_custom_prompt() False so
    # the official dataset prompt is used for every question (same choice as VsiBench / ReVSI).
    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='VSTIBench', pack=False, nframe=0, fps=-1, **kwargs):
        if dataset not in self.supported_datasets():
            raise ValueError(f'Unknown VSTI-Bench key {dataset}')
        if (nframe or 0) > 0 and int(nframe) != NUM_FRAMES:
            raise ValueError(f'{dataset}: the official protocol samples {NUM_FRAMES} frames; got nframe={nframe}')
        if fps is not None and fps > 0:
            raise ValueError(f'{dataset}: fps sampling is not supported (official {NUM_FRAMES}-frame budget)')
        self.num_frames = NUM_FRAMES
        super().__init__(dataset=dataset, pack=pack, nframe=NUM_FRAMES, fps=-1)

    @classmethod
    def supported_datasets(cls):
        return ['VSTIBench']

    def prepare_dataset(self, dataset):
        root = vstibench_root()
        tsv = os.path.join(LMUDataRoot(), f'{dataset}.tsv')
        manifest = os.path.join(LMUDataRoot(), f'{dataset}.manifest.json')
        os.makedirs(root, exist_ok=True)
        with portalocker.Lock(os.path.join(root, '.prepare.lock'), 'w', timeout=7200):
            problem = validate_cache(root, tsv, manifest)
            if problem is not None:
                print(f'[{dataset}] cache invalid ({problem}); rebuilding')
                build_cache(root, tsv, manifest)
                problem = validate_cache(root, tsv, manifest)
                if problem is not None:
                    raise RuntimeError(f'{dataset}: cache still invalid after rebuild: {problem}')
        return dict(root=root, data_file=tsv)

    def video_path(self, video):
        return os.path.join(self.data_root, video)

    def save_video_frames(self, video):
        """The official 32 frames (np.linspace(..., dtype=int)) as jpgs (non-VIDEO_LLM path)."""
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
            vid = decord.VideoReader(self.video_path(video))
            for i, pth in zip(official_frame_indices(len(vid)), frame_paths):
                if not os.path.exists(pth):
                    tmp = pth + '.tmp.jpg'
                    Image.fromarray(vid[i].asnumpy()).save(tmp)
                    os.replace(tmp, pth)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        qtype = line['question_type']
        question = line['question']
        # utils.vsibench_doc_to_text, verbatim
        if qtype in NA_QUESTION_TYPES:
            prompt = PRE_PROMPT + '\n' + question + '\n' + NA_POST_PROMPT
        elif qtype in MCA_QUESTION_TYPES:
            options = 'Options:\n' + '\n'.join(parse_options(line['options']))
            prompt = '\n'.join([PRE_PROMPT, question, options, MCA_POST_PROMPT])
        else:
            raise ValueError(f'Unknown question type: {qtype}')

        message = []
        if video_llm:
            # nframes pinned to the official budget (overrides every model-level sampling attribute).
            message.append(dict(type='video', value=self.video_path(line['video']), nframes=self.num_frames))
        else:
            for p in self.save_video_frames(line['video']):
                message.append(dict(type='image', value=p))
        message.append(dict(type='text', value=prompt))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        meta = self.data.set_index('index')
        for c in ('mc_answer', 'ground_truth', 'options', 'question_type'):
            data[c] = data['index'].map(meta[c])
        for c in ('mc_answer', 'options'):
            data[c] = data[c].where(data[c].notna(), '')
        scored, summary, table = score_dataframe(data)
        scope = evaluation_scope(data['index'], data['question_type'], meta.index)
        summary.update(scope)
        summary['n_zero_target'] = n_zero_targets(data['question_type'], data['ground_truth'])
        if scope['headline_partial']:
            warnings.warn(f'[{self.dataset_name}] partial input ({scope}); Overall is not the release metric')

        score_file = get_intermediate_file_path(eval_file, '_vstibench_score', 'xlsx')
        table_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(scored, score_file)
        table.to_csv(table_file, index=False)
        print(f'[{self.dataset_name}] per-task table ({table_file}):\n'
              + table.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        return summary
