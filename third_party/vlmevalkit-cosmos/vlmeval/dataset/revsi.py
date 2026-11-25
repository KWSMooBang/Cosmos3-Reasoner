"""ReVSI (ICML 2026): Rebuilding Visual Spatial Intelligence Evaluation.

Project page: https://3dlg-hcvc.github.io/revsi/   Data: HF 3dlg-hcvc/ReVSI (apache-2.0)

Ported from the official evaluation code (prompts, MCQ / numeric question types, MRA thresholds,
answer parsing, task merging, Avg):
  * open-compass/VLMEvalKit  vlmeval/dataset/revsi.py           @ f5296f04f3d6b5d5664f55aa64014c9dc942ed15
  * EvolvingLMMs-Lab/lmms-eval lmms_eval/tasks/revsi/utils.py   @ 9134731f4f617e62ee80bcae9d0d5d9695ea55ef
  * eamonn-zh/torchmetrics_ext metrics/vqa/{revsi,_spatial_vqa}.py @ 3c18d782f51cbe31558ea7b53cbece5561b1721e
  (3dlg-hcvc/revsi @ b26df855 points to these three; all three agree on prompt and scoring.)
Data pinned to HF revision 345a811321e2c9ec2b91a2e8df13e3a124584fa8.

Frames. The HF `video.zip` ships one pre-sampled mp4 per scene and frame budget
(`<N>_frame/<scene_id>.mp4`, exactly N frames, hierarchical uniform sampling of the all-frame
video; indices in metadata/sampled_video_frame_idx.json). The question set differs per budget
(questions whose objects are not visible in the N frames were removed), so every key reads its
own parquet subset. Every frame of the N-frame mp4 must reach the model:
  * VIDEO_LLM clients (CosmosReason*): the raw N-frame mp4 is sent as a video message WITHOUT
    sampling keys; the client samples with the model-level `nframes`, which the data config sets
    to N (`video_sampling.nframes`). qwen-vl-utils' linspace(0, N-1, N) on an N-frame video is
    the identity, so all N official frames are sent (scripts/check_revsi_frames.py proves it on
    the real files).
  * image-frame clients: all N frames of the mp4 are decoded and sent as images (upstream
    VLMEvalKit's `len/(n+1)` stepping on an N-frame video selects the same N frames).

Differences from the official code (all documented in PROTOCOL.md):
  * Keys ReVSI_{16,32,64}frame / ReVSI_tiny (official: revsi_{16,32,64,all}_frame). all_frame
    is not provided (no fixed frame budget). ReVSI_tiny = 64-frame subset restricted to
    metadata/tiny_set_question_ids.txt (the paper's 1,093-question set for proprietary models;
    1,057 of them are in the 64-frame subset).
  * TYPE = 'MCQ' (official: 'Video-VQA') so that CosmosReason*.use_custom_prompt() returns False
    and the official dataset prompt is used (same choice as VsiBench in this kit).
  * Scoring: `*_official` columns reproduce the official parser bit-for-bit (first
    whitespace token, rstrip('.'), case-insensitive letter match / float + MRA). The headline
    columns additionally fall back to the kit's rule extractors (can_match_option /
    can_match_na) ONLY when the official parse fails (not a letter / not a finite number), e.g.
    "(B)", "Answer: 3", "3m", "</think> about 2.5 meters". Whenever the official parse succeeds
    both scores are identical.
"""

import ast
import math
import os
import re
import zipfile
from collections import OrderedDict

import numpy as np
import pandas as pd
import portalocker

from vlmeval.smp import LMUDataRoot, dump, load
from vlmeval.smp.file import get_intermediate_file_path
from .video_base import VideoBaseDataset

HF_REPO = '3dlg-hcvc/ReVSI'
HF_REVISION = '345a811321e2c9ec2b91a2e8df13e3a124584fa8'
# sha256 of video.zip @ HF_REVISION (= its Git LFS oid on the Hub).
VIDEO_ZIP_SHA256 = '7d10f8a538b1a03e83c8c01b9c11a595db6b49d661e22f2dc513358f8eabb9fe'

PRE_PROMPT = 'These are frames of a video.'
NQ_POST_PROMPT = 'Answer the question using a single integer or decimal number.'
MCQ_POST_PROMPT = "Answer with the option's letter from the given choices directly."

NQ_QUESTION_TYPES = [
    'object_counting_single',
    'object_counting_multiple',
    'object_abs_distance',
    'object_size_estimation',
    'room_size_estimation_single',
    'room_size_estimation_multiple',
]
MCQ_QUESTION_TYPES = [
    'object_rel_direction_forward_easy',
    'object_rel_direction_backward_easy',
    'object_rel_direction_forward_hard',
    'object_rel_direction_backward_hard',
    'object_rel_distance_closest',
    'object_rel_distance_farthest',
    'route_planning',
]
# Reported task -> question types averaged into it (mean of sub-type means, as upstream).
# Order = paper table order (Obj. Cnt, Abs. Dist, Obj. Size, Room Size, Rel. Dist, Rel. Dir, Route Plan).
TASKS = OrderedDict([
    ('object_counting', ['object_counting_single', 'object_counting_multiple']),
    ('object_abs_distance', ['object_abs_distance']),
    ('object_size_estimation', ['object_size_estimation']),
    ('room_size_estimation', ['room_size_estimation_single', 'room_size_estimation_multiple']),
    ('object_rel_distance', ['object_rel_distance_closest', 'object_rel_distance_farthest']),
    ('object_rel_direction', ['object_rel_direction_forward_easy', 'object_rel_direction_backward_easy',
                              'object_rel_direction_forward_hard', 'object_rel_direction_backward_hard']),
    ('route_planning', ['route_planning']),
])
TASK_LABELS = {
    'object_counting': 'Obj. Count', 'object_abs_distance': 'Abs. Dist', 'object_size_estimation': 'Obj. Size',
    'room_size_estimation': 'Room Size', 'object_rel_distance': 'Rel. Dist', 'object_rel_direction': 'Rel. Dir',
    'route_planning': 'Route Plan',
}

# dataset key -> (frame budget, tiny-set only)
SUBSETS = {
    'ReVSI_16frame': (16, False),
    'ReVSI_32frame': (32, False),
    'ReVSI_64frame': (64, False),
    'ReVSI_tiny': (64, True),
}


# ----------------------------------------------------------------------------------------------
# Scoring (pure functions; usable without the videos, see scripts/test_revsi_scoring.py)
# ----------------------------------------------------------------------------------------------
def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """Official MRA: thresholds linspace(0.5, 0.95, int((end-start)/interval + 2)) = 0.50..0.95 (10)."""
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = (abs(pred - target) / target) <= (1 - conf_intervs)
    return float(accuracy.mean())


def official_token(prediction) -> str:
    """Official answer normalisation: first whitespace token, trailing '.' removed."""
    return str(prediction).strip().split(' ')[0].rstrip('.').strip()


def official_score(question_type, prediction, ground_truth) -> float:
    """Bit-for-bit port of upstream VLMEvalKit ReVSI.evaluate / lmms-eval revsi_process_results."""
    pred_answer = official_token(prediction)
    gt_answer = str(ground_truth)
    if question_type in MCQ_QUESTION_TYPES:
        return 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
    if question_type in NQ_QUESTION_TYPES:
        try:
            return mean_relative_accuracy(float(pred_answer), float(gt_answer))
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    raise ValueError(f'Unknown ReVSI question type: {question_type}')


def _option_letters(options) -> str:
    opts = parse_options(options)
    letters = ''.join(str(o).strip()[:1].upper() for o in opts if str(o).strip())
    return letters or 'ABCD'


_THOUSANDS_RE = re.compile(r'(?<=\d),(?=\d{3}(?!\d))')


def robust_parse(question_type, prediction, options=None):
    """(parsed answer or None, mode) with mode in {'official', 'fallback', 'fail'}.

    The official token is used whenever it is a valid answer; only otherwise the kit's rule
    extractors are tried on the full text (after the last </think> if present).
    """
    from .utils.spatial_bench.matching_func import can_match_na, can_match_option

    text = '' if prediction is None or (isinstance(prediction, float) and math.isnan(prediction)) \
        else str(prediction)
    tok = official_token(text)
    if question_type in MCQ_QUESTION_TYPES:
        letters = _option_letters(options)
        if len(tok) == 1 and tok.upper() in letters:
            return tok.upper(), 'official'
        letter = can_match_option(text, choices=list(letters)) if text.strip() else False
        if letter and letter in letters:
            return letter, 'fallback'
        return None, 'fail'
    if question_type in NQ_QUESTION_TYPES:
        try:
            v = float(tok)
            if math.isfinite(v):
                return v, 'official'
        except (TypeError, ValueError):
            pass
        v = can_match_na(_THOUSANDS_RE.sub('', text)) if text.strip() else None
        if v is not None and math.isfinite(v):
            return v, 'fallback'
        return None, 'fail'
    raise ValueError(f'Unknown ReVSI question type: {question_type}')


def robust_score(question_type, prediction, ground_truth, options=None):
    parsed, mode = robust_parse(question_type, prediction, options)
    if parsed is None:
        return 0.0, parsed, mode
    if question_type in MCQ_QUESTION_TYPES:
        return (1.0 if str(parsed).lower() == str(ground_truth).strip().lower() else 0.0), parsed, mode
    try:
        return mean_relative_accuracy(float(parsed), float(ground_truth)), parsed, mode
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0, parsed, mode


def _aggregate(df: pd.DataFrame, col: str) -> OrderedDict:
    """Per question-type mean -> merged task means (mean of sub-type means) -> Avg over present tasks."""
    per_qtype = df.groupby('question_type')[col].mean().to_dict()
    out = OrderedDict()
    for task, qtypes in TASKS.items():
        vals = [per_qtype[q] for q in qtypes if q in per_qtype]
        if vals:
            out[task] = float(np.mean(vals))
    return out


def score_dataframe(df: pd.DataFrame):
    """Score a prediction table (columns question_type, prediction, answer|ground_truth, options).

    Returns (scored per-row DataFrame, summary dict in 0-100, per-task table DataFrame).
    """
    df = df.copy()
    gt_col = 'ground_truth' if 'ground_truth' in df.columns else 'answer'
    if 'options' not in df.columns:
        df['options'] = None
    off, rob, parsed, modes = [], [], [], []
    for _, r in df.iterrows():
        pred = r['prediction']
        gt = r[gt_col]
        off.append(official_score(r['question_type'], pred, gt))
        s, p, m = robust_score(r['question_type'], pred, gt, r['options'])
        rob.append(s)
        parsed.append(p)
        modes.append(m)
    df['score_official'] = off
    df['score'] = rob
    df['pred_parsed'] = parsed
    df['parse_mode'] = modes

    tasks_rob = _aggregate(df, 'score')
    tasks_off = _aggregate(df, 'score_official')
    summary = OrderedDict()
    summary['Overall'] = 100.0 * float(np.mean(list(tasks_rob.values()))) if tasks_rob else 0.0
    summary['Overall_official'] = 100.0 * float(np.mean(list(tasks_off.values()))) if tasks_off else 0.0
    for task in tasks_rob:
        summary[task] = 100.0 * tasks_rob[task]
    for task in tasks_off:
        summary[f'{task}_official'] = 100.0 * tasks_off[task]

    rows = []
    task_of = {q: t for t, qs in TASKS.items() for q in qs}
    df['task'] = df['question_type'].map(task_of)
    for task in tasks_rob:
        sub = df[df['task'] == task]
        rows.append(dict(
            task=TASK_LABELS[task], key=task,
            metric='MRA' if TASKS[task][0] in NQ_QUESTION_TYPES else 'Acc',
            n=len(sub), score=100.0 * tasks_rob[task], score_official=100.0 * tasks_off[task],
            n_fallback=int((sub['parse_mode'] == 'fallback').sum()),
            n_unparsed=int((sub['parse_mode'] == 'fail').sum()),
        ))
    rows.append(dict(
        task='Avg', key='overall', metric='mean of tasks', n=len(df),
        score=summary['Overall'], score_official=summary['Overall_official'],
        n_fallback=int((df['parse_mode'] == 'fallback').sum()),
        n_unparsed=int((df['parse_mode'] == 'fail').sum()),
    ))
    table = pd.DataFrame(rows)
    return df, summary, table


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
        return [str(o) for o in ast.literal_eval(s)]
    return [str(options)]


def _serialize_options(options):
    opts = parse_options(options)
    return repr(opts) if opts else ''


# ----------------------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------------------
def revsi_root() -> str:
    return os.path.join(LMUDataRoot(), 'ReVSI')


def _download(root, patterns):
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type='dataset', revision=HF_REVISION, local_dir=root,
                      allow_patterns=patterns)


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


def _ensure_video_zip(root, zip_path):
    """video.zip present and sha256 == LFS oid; otherwise delete and re-download (once)."""
    for attempt in range(2):
        if not os.path.exists(zip_path):
            _download(root, ['video.zip'])
        digest = _sha256(zip_path)
        if digest == VIDEO_ZIP_SHA256:
            return
        os.remove(zip_path)
        # forget snapshot_download's local-dir metadata so the file is fetched again
        meta = os.path.join(root, '.cache', 'huggingface', 'download', 'video.zip.metadata')
        if os.path.exists(meta):
            os.remove(meta)
    raise RuntimeError(f'{zip_path}: sha256 {digest} != {VIDEO_ZIP_SHA256} after re-download')


def _missing_videos(video_root, videos):
    """Relative video paths that do not exist or are empty."""
    out = []
    for v in videos:
        p = os.path.join(video_root, v)
        if not (os.path.isfile(p) and os.path.getsize(p) > 0):
            out.append(v)
    return out


def _extract_subset(zip_path, video_root, subset):
    """Extract only `<subset>/*.mp4` members of video.zip into video_root/<subset>/ (path-safe)."""
    dst_dir = os.path.abspath(os.path.join(video_root, subset))
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = os.path.normpath(info.filename).replace('\\', '/').split('/')
            if info.is_dir() or subset not in parts[:-1] or not parts[-1].endswith('.mp4'):
                continue
            dst = os.path.abspath(os.path.join(dst_dir, parts[-1]))
            if not dst.startswith(dst_dir + os.sep):
                raise RuntimeError(f'Unsafe path in zip: {info.filename}')
            os.makedirs(dst_dir, exist_ok=True)
            with zf.open(info) as src, open(dst + '.tmp', 'wb') as out:
                while True:
                    buf = src.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)
            os.replace(dst + '.tmp', dst)
            n += 1
    if n == 0:
        raise RuntimeError(f'No {subset}/*.mp4 member in {zip_path}')
    return n


class ReVSI(VideoBaseDataset):
    # 'MCQ' (not the official 'Video-VQA'): keeps CosmosReason*.use_custom_prompt() False so the
    # official dataset prompt below is used for every question (VsiBench does the same).
    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='ReVSI_64frame', pack=False, nframe=0, fps=-1, **kwargs):
        if dataset not in SUBSETS:
            raise ValueError(f'Unknown ReVSI key {dataset}; choose from {list(SUBSETS)}')
        self.num_frames, self.tiny = SUBSETS[dataset]
        self.frame_subset = f'{self.num_frames}_frame'
        if (nframe or 0) > 0 and int(nframe) != self.num_frames:
            raise ValueError(f'{dataset} uses the official pre-sampled {self.num_frames}-frame videos; '
                             f'nframe={nframe} would not match the paper protocol')
        if fps is not None and fps > 0:
            raise ValueError(f'{dataset}: fps sampling is not supported (fixed {self.num_frames}-frame budget)')
        super().__init__(dataset=dataset, pack=pack, nframe=self.num_frames, fps=-1)

    @classmethod
    def supported_datasets(cls):
        return list(SUBSETS)

    def prepare_dataset(self, dataset):
        root = revsi_root()
        subset = self.frame_subset
        parquet = os.path.join(root, subset, 'test-00000-of-00001.parquet')
        tiny_ids = os.path.join(root, 'metadata', 'tiny_set_question_ids.txt')
        zip_path = os.path.join(root, 'video.zip')
        video_root = os.path.join(root, 'videos')
        sentinel = os.path.join(video_root, f'.{subset}.extracted')
        os.makedirs(root, exist_ok=True)

        with portalocker.Lock(os.path.join(root, '.prepare.lock'), 'w', timeout=3600):
            if not (os.path.exists(parquet) and os.path.exists(tiny_ids)):
                _download(root, [f'{subset}/*', 'metadata/*', 'README.md'])
            # Every mp4 this subset references must exist and be non-empty; the marker alone is
            # not trusted (partial extraction, deleted files).
            scenes = sorted(set(pd.read_parquet(parquet, columns=['scene_id'])['scene_id'].astype(str)))
            needed = [f'{subset}/{sc}.mp4' for sc in scenes]
            if not os.path.exists(sentinel) or _missing_videos(video_root, needed):
                if os.path.exists(sentinel):
                    os.remove(sentinel)
                _ensure_video_zip(root, zip_path)
                n = _extract_subset(zip_path, video_root, subset)
                missing = _missing_videos(video_root, needed)
                if missing:
                    raise RuntimeError(f'{len(missing)} {subset} videos missing after extraction, e.g. {missing[:3]}')
                with open(sentinel, 'w') as f:
                    f.write(f'{n}\n')

            tsv = os.path.join(LMUDataRoot(), f'{dataset}.tsv')
            if not os.path.exists(tsv):
                df = pd.read_parquet(parquet)
                if self.tiny:
                    with open(tiny_ids) as f:
                        ids = {int(x) for x in f.read().split()}
                    df = df[df['id'].isin(ids)]
                assert set(df['num_frames'].astype(str)) == {str(self.num_frames)}, 'subset/num_frames mismatch'
                out = pd.DataFrame({
                    'index': df['id'].astype(int),
                    'id': df['id'].astype(int),
                    'source': df['dataset'],
                    'scene_id': df['scene_id'].astype(str),
                    'question_type': df['question_type'],
                    'question': df['question'],
                    'options': df['options'].apply(_serialize_options),
                    'answer': df['ground_truth'].astype(str),
                    'video': [f'{subset}/{s}.mp4' for s in df['scene_id'].astype(str)],
                })
                tmp = tsv + '.tmp.tsv'
                out.to_csv(tmp, sep='\t', index=False)
                os.replace(tmp, tsv)
        return dict(root=video_root, data_file=tsv)

    def video_path(self, video):
        return os.path.join(self.data_root, video)

    def save_video_frames(self, video):
        """All N frames of the pre-sampled N-frame mp4, as jpgs (non-VIDEO_LLM path)."""
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
            if len(vid) != self.num_frames:
                raise RuntimeError(f'{video}: {len(vid)} frames, expected the official {self.num_frames}')
            for i, pth in enumerate(frame_paths):
                if not os.path.exists(pth):
                    Image.fromarray(vid[i].asnumpy()).save(pth)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        qtype = line['question_type']
        question = line['question']
        if qtype in NQ_QUESTION_TYPES:
            prompt = '\n'.join([PRE_PROMPT, question, NQ_POST_PROMPT]).strip()
        elif qtype in MCQ_QUESTION_TYPES:
            options = 'Options:\n' + '\n'.join(parse_options(line['options']))
            prompt = '\n'.join([PRE_PROMPT, question, options, MCQ_POST_PROMPT]).strip()
        else:
            raise ValueError(f'Unknown ReVSI question type: {qtype}')

        message = []
        if video_llm:
            # No sampling keys here on purpose: the client's model-level `nframes` (= N, from the
            # data config's video_sampling) must apply. See module docstring.
            message.append(dict(type='video', value=self.video_path(line['video'])))
        else:
            for p in self.save_video_frames(line['video']):
                message.append(dict(type='image', value=p))
        message.append(dict(type='text', value=prompt))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        scored, summary, table = score_dataframe(data)

        score_file = get_intermediate_file_path(eval_file, '_revsi_score', 'xlsx')
        table_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(scored, score_file)
        table.to_csv(table_file, index=False)
        cols = ['task', 'metric', 'n', 'score', 'score_official', 'n_fallback', 'n_unparsed']
        print(f'[{self.dataset_name}] per-task table ({table_file}):\n'
              + table[cols].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        return summary
