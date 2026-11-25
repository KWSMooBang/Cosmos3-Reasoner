"""QuantiPhy (CVPR 2026): quantitative physical reasoning from video in real-world units.

Project page: https://quantiphy.stanford.edu/   Paper: arXiv 2512.19526
Data (CC BY 4.0 annotations; videos keep their original licences):
  * HF PaulineLi/QuantiPhy-validation @ 2d56ff31c22c12bf377b2c2092ff229d291fd830  (158 QA, GT released, 480p videos)
  * HF PaulineLi/QuantiPhy            @ ca0658a1ffba930b8170223db7a377cc80192bd6  (3,289 QA "official test set",
    GT withheld; videos at original resolution only)
Official code (MIT): github.com/Paulineli/QuantiPhy @ aa8daa7a5736ce3514ed985189ad27ef4a2e6e68
  * model_run_example/run_API_results.py (sha256 53f5d8c7...): SYSTEM_PROMPT, build_context_prefix,
    zero-shot v3 user text, EXACT_NUMBER_WITH_UNIT_RE / num_re / parse_number  -> ported verbatim below
  * evaluator.py (sha256 98c741f5...): MRA over C = {0.1..0.9, 0.95}, category = inference_type[0] +
    video_type[1] (S2, D2, S3, D3), score = unweighted mean of the four category means; bg / obj
    breakdowns (segmentation clips count as background X); invalid_percentage -> ported verbatim below

Keys
  * QuantiPhy       validation split (158 QA, 24 videos): the only split with public GT -> scored.
  * QuantiPhy_test  official test split (3,289 QA, 568 videos): inference only. evaluate() writes the
    official submission CSV (template columns, ids matched to the template by question identity) and reports the
    parse-failure rate; no score (GT withheld, the challenge portal is closed).

Request (official zero-shot v3 = the paper's main setting):
  system: SYSTEM_PROMPT (verbatim)
  user:   [all frames of the 480p video] + "Given that {prior}. [Additionally, ... {depth_info}] {question}
          \n\nPlease answer the question with numbers and units ONLY. No explanation needed."
  depth_info is included only for 3D video types (video_type[1] == '3'), as in the official script.

Frames. Official: every frame of the 480p video (cv2, no sampling), sent as JPEG images.
  * VIDEO_LLM clients (CosmosReason*): the 480p mp4 is sent as one video message with
    nframes = 2 * floor(n_frames / 2) (n_frames counted with decord, stored in the TSV). qwen-vl-utils
    requires an even nframes <= n_frames, so odd-length clips lose one frame (linspace over all
    frames); every other frame is sent, at the clip's own fps (timestamps intact). Setting nframes in
    the message disables all model-level sampling keys (CosmosReason._get_video_processing_kwargs).
    `max_pixels` is NOT set: 480p frames are below the qwen-vl-utils per-frame cap, so they are sent
    at native 480p (rounded to multiples of 32); total_pixels is the qwen-vl-utils default.
  * Other clients: every frame is decoded and sent as an image (identical to the official request).
  The server-side Qwen3-VL video processor then caps T*H*W at its size.longest_edge (25,165,824 px for
  the Cosmos3-Nano checkpoint), i.e. long clips are downscaled below 480p there (all frames kept).
  scripts/check_quantiphy_frames.py proves the frame counts on the real files;
  scripts/quantiphy_token_budget.py reports the resulting grid and prompt tokens with the real processor.

Differences from the official code (PROTOCOL.md, QuantiPhy):
  * Frames go as one video (Qwen3-VL merges frame pairs temporally; server-side pixel budget) instead
    of N separate images.
  * No re-ask on unparseable replies (official: up to 5 attempts). With greedy decoding (T=0, fixed
    seed) the repeat would be identical; failures score 0 as in evaluator.py.
  * QuantiPhy_test videos: our own 480p transcodes (ffmpeg, short side 480, every frame kept), because
    the test repo ships only original-resolution videos; the validation repo ships the official 480p files.
  * Headline keys use the official parser bit-for-bit, including its "^2" quirk ("980 cm/s^2" -> 2.0,
    see _EXP_RE). Diagnostic `*_lenient` keys normalise the exponent first and convert the reply's unit
    to the unit asked in the question ("0.25 m" for a question in cm -> 25); the official parser strips
    units without conversion.
"""

import math
import os
import re
import subprocess
from collections import OrderedDict

import numpy as np
import pandas as pd
import portalocker

from vlmeval.smp import LMUDataRoot, dump, load
from vlmeval.smp.file import get_intermediate_file_path
from .video_base import VideoBaseDataset

VAL_REPO = 'PaulineLi/QuantiPhy-validation'
VAL_REVISION = '2d56ff31c22c12bf377b2c2092ff229d291fd830'
TEST_REPO = 'PaulineLi/QuantiPhy'
TEST_REVISION = 'ca0658a1ffba930b8170223db7a377cc80192bd6'
OFFICIAL_CODE = 'github.com/Paulineli/QuantiPhy@aa8daa7a5736ce3514ed985189ad27ef4a2e6e68'
TEMPLATE_URL = 'https://quantiphy.stanford.edu/competition/eval/quantiphy_submission_template.csv'
TEMPLATE_SHA256 = '5f1e8367308ebaee6718929d74a175fecd6813a8d5862fc72a709d07d2328c18'
SUBMISSION_COLUMNS = ['id', 'video_id', 'video_source', 'video_type', 'fps', 'inference_type', 'question',
                      'ground_truth_prior', 'depth_info', 'parsed_value']

SUBSETS = {'QuantiPhy': 'validation', 'QuantiPhy_test': 'test'}
CATEGORIES = ['S2', 'D2', 'S3', 'D3']

# ---------------------------------------------------------------------------------------------
# Official prompt (run_API_results.py, zero_shot_prompt version 3 and build_context_prefix)
# ---------------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in physics measurements.\n"
    "Analyze the video frames carefully and provide ONLY the numerical answer with units. No explanation or reasoning needed.\n"  # noqa: E501
    "Format your response as: [value] [unit]\n"
    "Example: 2.5 cm\n"
    "Be as accurate as possible with measurements and calculations. Please give me an estimated answer even if you are not sure."  # noqa: E501
)
POST_PROMPT = "\n\nPlease answer the question with numbers and units ONLY. No explanation needed."


def _isna(x):
    return x is None or (isinstance(x, float) and math.isnan(x)) or (not isinstance(x, str) and pd.isna(x))


def build_context_prefix(prior=None, depth_info=None):
    """Verbatim port of run_API_results.build_context_prefix."""
    parts = []

    if prior is not None and not pd.isna(prior):
        prior_text = str(prior).strip()
        if prior_text:
            parts.append(f"Given that {prior_text}.")

    if depth_info is not None and not pd.isna(depth_info):
        depth_text = str(depth_info).strip()
        if depth_text:
            parts.append(
                "Additionally, you have the following information about the distance between the objects in the video and the shooting camera: "  # noqa: E501
                f"{depth_text}"
            )

    if not parts:
        return ""

    context = " ".join(parts).strip()
    if context[-1] not in ".!?":
        context += "."
    return context + " "


def build_user_text(question, prior, depth_info, video_type):
    """Official zero-shot v3 user text; depth only for 3D video types (official include_depth)."""
    video_type_value = str(video_type if not _isna(video_type) else '')
    include_depth = len(video_type_value) >= 2 and video_type_value[1] == '3'
    depth = depth_info if include_depth else None
    return f"{build_context_prefix(prior, depth)}{question}{POST_PROMPT}"


# ---------------------------------------------------------------------------------------------
# Official answer parsing (run_API_results.py "2. Number Parsing"), verbatim
# ---------------------------------------------------------------------------------------------
num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
EXACT_NUMBER_WITH_UNIT_RE = re.compile(
    r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(?:"
    r"meters?|meter|m|kilometers?|kilometer|km|centimeters?|centimeter|cm|millimeters?|millimeter|mm|"
    r"inches?|inch|in|feet|foot|ft|yards?|yard|yd|"
    r"seconds?|second|s|minutes?|minute|min|hours?|hour|h|"
    r"m/s|m/s²|m/s^2|m/s2|km/h|kph|mph|ft/s|cm/s|mm/s|"
    r"m/s/s|m/s²|m/s^2|m/s2|ft/s²|ft/s^2|ft/s2|cm/s²|cm/s^2|cm/s2|mm/s²|mm/s^2|mm/s2|g"
    r")?\s*$",
    re.IGNORECASE,
)


def parse_number(text: str) -> float:
    """Verbatim port of run_API_results.parse_number (absolute value, NaN on failure)."""
    if not text or pd.isna(text):
        return np.nan

    original_text = str(text).strip()

    # Step 1: Exact Match Validation
    if EXACT_NUMBER_WITH_UNIT_RE.match(original_text):
        num_match = num_re.search(original_text)
        if num_match:
            try:
                value = float(num_match.group(0))
                return abs(value)
            except ValueError:
                pass

    # Step 2: Delimiter Search
    text_to_parse = original_text
    delimiters = ["Final Answer:", "Answer:", "=>", "=", ":", "is:", "The answer is:", "Example:"]

    last_delimiter_pos = -1
    last_delimiter = None
    for delimiter in delimiters:
        positions = [m.start() for m in re.finditer(re.escape(delimiter), text_to_parse, re.IGNORECASE)]
        if positions:
            last_pos = max(positions)
            if last_pos > last_delimiter_pos:
                last_delimiter_pos = last_pos
                last_delimiter = delimiter

    if last_delimiter_pos >= 0 and last_delimiter:
        parts = text_to_parse.rsplit(last_delimiter, 1)
        if len(parts) > 1:
            text_to_parse = parts[-1].strip()

    # Step 3: Unit Sanitization
    text_cleaned = text_to_parse.lower()
    units_to_remove = [
        # size / distance
        'kilometers', 'kilometer', 'km',
        'meters', 'meter', 'm',
        'centimeters', 'centimeter', 'cm',
        'millimeters', 'millimeter', 'mm',
        'inches', 'inch', 'in',
        'feet', 'foot', 'ft',
        'yards', 'yard', 'yd',
        # time
        'hours', 'hour', 'h',
        'minutes', 'minute', 'min',
        'seconds', 'second', 's',
        # speed
        'km/h', 'kph', 'mph', 'ft/s', 'm/s', 'cm/s', 'mm/s',
        # acceleration
        'm/s/s', 'm/s²', 'm/s^2', 'm/s2',
        'ft/s²', 'ft/s^2', 'ft/s2',
        'cm/s²', 'cm/s^2', 'cm/s2',
        'mm/s²', 'mm/s^2', 'mm/s2',
        'g',
        # area (existing support retained)
        'm^2', 'm²',
    ]
    for unit in units_to_remove:
        text_cleaned = text_cleaned.replace(unit, '')

    # Step 4: Heuristic Extraction
    all_numbers = num_re.findall(text_cleaned)

    if all_numbers:
        try:
            value = float(all_numbers[-1])
            return abs(value)
        except (ValueError, IndexError):
            pass

    all_numbers_original = num_re.findall(original_text)
    if all_numbers_original:
        try:
            value = float(all_numbers_original[-1])
            return abs(value)
        except (ValueError, IndexError):
            pass

    # Step 5: Failure Handling
    return np.nan


# ---------------------------------------------------------------------------------------------
# Official scoring (evaluator.py)
# ---------------------------------------------------------------------------------------------
MRA_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def mra(parsed_value, gt) -> float:
    """evaluator.py row MRA: |abs(pred) - gt| / gt < 1 - theta over 10 thresholds; NaN if gt is NaN or 0."""
    gt = pd.to_numeric(gt, errors='coerce')
    if not (pd.notna(gt) and gt != 0):
        return float('nan')
    p = pd.to_numeric(parsed_value, errors='coerce')
    p = abs(p) if pd.notna(p) else float('nan')
    return sum(abs(p - gt) / gt < (1 - theta) for theta in MRA_THRESHOLDS) / 10


def category_of(inference_type, video_type):
    return inference_type[0] + video_type[1] if pd.notna(video_type) else None


def invalid_count(parsed):
    """evaluator.py invalid_percentage numerator: blank + non-numerical + zero parsed values."""
    s = pd.Series(parsed, dtype=object)
    num = pd.to_numeric(s, errors='coerce')
    return int(s.isna().sum() + (s.notna() & num.isna()).sum() + (num == 0).sum())


def _four_cat_mean(df, col):
    m = {c: df.loc[df['category'] == c, col].mean() for c in CATEGORIES if (df['category'] == c).any()}
    return (sum(m[c] for c in CATEGORIES) / 4) if len(m) == 4 else None, m


def score_frame(df: pd.DataFrame, parsed_col='parsed_value', gt_col='answer', mra_col='mra'):
    """Per-row MRA + official aggregates (fractions in [0, 1], as evaluator.py)."""
    df = df.copy()
    if 'video_source' in df.columns:  # official: segmentation clips count as plain background
        mask = df['video_source'].isin(['segmentation', 'segmentaion'])
        df.loc[mask, 'video_type'] = df.loc[mask, 'video_type'].apply(
            lambda x: x[:-1] + 'X' if pd.notna(x) and len(x) > 0 else x)
    df['category'] = [category_of(i, v) for i, v in zip(df['inference_type'], df['video_type'])]
    df[mra_col] = [mra(p, g) for p, g in zip(df[parsed_col], df[gt_col])]
    avg, per_cat = _four_cat_mean(df, mra_col)
    out = OrderedDict(mra_average=avg)
    for c in CATEGORIES:
        out[f'mra_{c}'] = per_cat.get(c)
    df['bg'] = df['video_type'].apply(lambda x: x[-1] if pd.notna(x) and len(x) > 0 else None)
    df['obj'] = df['video_type'].apply(lambda x: {'S': 'single', 'M': 'multiple'}.get(x[2]) if pd.notna(x)
                                       and len(x) > 2 else None)
    for key, col in (('bg', 'bg'), ('obj', 'obj')):
        for v in sorted(df[col].dropna().unique()):
            out[f'mra_{key}_{v}'] = _four_cat_mean(df[df[col] == v], mra_col)[0]
    out['invalid_percentage'] = invalid_count(df[parsed_col]) / len(df) * 100 if len(df) else 0
    return df, out


# ---------------------------------------------------------------------------------------------
# Diagnostic unit conversion (not part of the official protocol)
# ---------------------------------------------------------------------------------------------
_LEN = {'km': 1000.0, 'm': 1.0, 'cm': 0.01, 'mm': 0.001, 'in': 0.0254, 'ft': 0.3048, 'yd': 0.9144}
_UNIT_ALIASES = [
    # (regex, canonical, dimension); longest first
    (r'km\s*/\s*h(?:r|our)?|kph|kilometers?\s+per\s+hour', 'km/h', 'vel'),
    (r'mph|miles?\s+per\s+hour', 'mph', 'vel'),
    (r'(?:mm|cm|m|ft)\s*/\s*s(?:ec)?\s*(?:\^\s*2|²|2|/\s*s)|(?:milli|centi)?met(?:er|re)s?\s+per\s+second\s+squared',
     'acc', 'acc'),
    (r'(?:mm|cm|m|ft)\s*/\s*s(?:ec)?|(?:milli|centi)?met(?:er|re)s?\s+per\s+second', 'vel', 'vel'),
    (r'kilomet(?:er|re)s?|km', 'km', 'len'),
    (r'centimet(?:er|re)s?|cm', 'cm', 'len'),
    (r'millimet(?:er|re)s?|mm', 'mm', 'len'),
    (r'met(?:er|re)s?|m', 'm', 'len'),
    (r'inch(?:es)?|in', 'in', 'len'),
    (r'feet|foot|ft', 'ft', 'len'),
    (r'yards?|yd', 'yd', 'len'),
]
_PREFIX_SCALE = {'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'ft': 0.3048, 'milli': 0.001, 'centi': 0.01}


def _unit_scale(unit_text):
    """unit text -> (dimension, SI scale) or None."""
    u = unit_text.strip().lower()
    for pat, canon, dim in _UNIT_ALIASES:
        if re.fullmatch(pat, u):
            if dim == 'len':
                return dim, _LEN[canon]
            if canon == 'km/h':
                return dim, 1000.0 / 3600.0
            if canon == 'mph':
                return dim, 1609.344 / 3600.0
            head = re.match(r'(mm|cm|ft|m)\b|(milli|centi)', u)
            key = (head.group(1) or head.group(2)) if head else 'm'
            return dim, _PREFIX_SCALE.get(key, 1.0)
    return None


_UNIT_TOKEN = '|'.join(p for p, _, _ in _UNIT_ALIASES)
_Q_UNIT_RE = re.compile(r'\b(?:in|\()\s*(' + _UNIT_TOKEN + r')\s*\)?\s*[?.]?\s*$', re.IGNORECASE)


def question_unit(question):
    m = _Q_UNIT_RE.search(str(question).strip())
    return _unit_scale(m.group(1)) if m else None


# "/s^2", "/s2", "/sec^2" -> "/s²". The official EXACT_NUMBER_WITH_UNIT_RE has an unescaped "^" in
# "m/s^2" (never matches) and its unit stripping removes "m"/"s" before "m/s^2", so the exponent "2"
# becomes the last number: "980 cm/s^2" -> 2.0 officially. "²" is not a \d digit, so it is harmless.
_EXP_RE = re.compile(r'(/\s*s(?:ec)?)\s*(?:\^\s*2|2)(?![\d.])', re.IGNORECASE)


# parse_number's unit list (copied; parse_number itself keeps its verbatim local list)
_UNITS_TO_REMOVE = [
    'kilometers', 'kilometer', 'km', 'meters', 'meter', 'm', 'centimeters', 'centimeter', 'cm',
    'millimeters', 'millimeter', 'mm', 'inches', 'inch', 'in', 'feet', 'foot', 'ft', 'yards', 'yard', 'yd',
    'hours', 'hour', 'h', 'minutes', 'minute', 'min', 'seconds', 'second', 's',
    'km/h', 'kph', 'mph', 'ft/s', 'm/s', 'cm/s', 'mm/s',
    'm/s/s', 'm/s²', 'm/s^2', 'm/s2', 'ft/s²', 'ft/s^2', 'ft/s2', 'cm/s²', 'cm/s^2', 'cm/s2',
    'mm/s²', 'mm/s^2', 'mm/s2', 'g', 'm^2', 'm²',
]
_PARSE_DELIMITERS = ["Final Answer:", "Answer:", "=>", "=", ":", "is:", "The answer is:", "Example:"]


def _remove_tracked(chars, idx, old):
    """str.replace(old, '') on ''.join(chars) (non-overlapping, left to right), keeping source indices."""
    s = ''.join(chars)
    out_c, out_i, i = [], [], 0
    while True:
        j = s.find(old, i)
        if j < 0:
            break
        out_c.extend(chars[i:j])
        out_i.extend(idx[i:j])
        i = j + len(old)
    out_c.extend(chars[i:])
    out_i.extend(idx[i:])
    return out_c, out_i


def parse_number_span(text):
    """Mirror of parse_number that also returns the (start, end) span, in str(text).strip(), of the
    characters the selected number came from (None if NaN). Value parity with parse_number is tested;
    lenient_value only trusts the span when both values agree."""
    if not text or pd.isna(text):
        return np.nan, None
    original_text = str(text).strip()
    if EXACT_NUMBER_WITH_UNIT_RE.match(original_text):
        m = num_re.search(original_text)
        if m:
            try:
                return abs(float(m.group(0))), (m.start(), m.end())
            except ValueError:
                pass
    text_to_parse, seg_start = original_text, 0
    last_pos, last_delim = -1, None
    for delimiter in _PARSE_DELIMITERS:
        positions = [m.start() for m in re.finditer(re.escape(delimiter), text_to_parse, re.IGNORECASE)]
        if positions and max(positions) > last_pos:
            last_pos, last_delim = max(positions), delimiter
    if last_pos >= 0 and last_delim:
        parts = text_to_parse.rsplit(last_delim, 1)  # case-sensitive, as the official code
        if len(parts) > 1:
            k = text_to_parse.rfind(last_delim) + len(last_delim)
            after = text_to_parse[k:]
            seg_start = k + len(after) - len(after.lstrip())
            text_to_parse = after.strip()
    chars, idx = [], []
    for i, c in enumerate(text_to_parse):
        for lc in c.lower():
            chars.append(lc)
            idx.append(seg_start + i)
    for unit in _UNITS_TO_REMOVE:
        chars, idx = _remove_tracked(chars, idx, unit)
    ms = list(num_re.finditer(''.join(chars)))
    if ms:
        try:
            m = ms[-1]
            return abs(float(m.group(0))), (idx[m.start()], idx[m.end() - 1] + 1)
        except (ValueError, IndexError):
            pass
    ms = list(num_re.finditer(original_text))
    if ms:
        try:
            return abs(float(ms[-1].group(0))), (ms[-1].start(), ms[-1].end())
        except (ValueError, IndexError):
            pass
    return np.nan, None


_UNIT_AFTER_RE = re.compile(r'\s*(' + _UNIT_TOKEN + r')(?![a-z])', re.IGNORECASE)


def lenient_value(prediction, question):
    """Diagnostic parse: official parse_number after normalising the acceleration exponent ("^2" -> "²"),
    then rescaled to the question's unit when the unit written directly after the number occurrence the
    parser selected (parse_number_span) differs. No unit after that occurrence -> no conversion."""
    if _isna(prediction):
        return parse_number(prediction)
    text = _EXP_RE.sub(r'\1²', str(prediction))
    v = parse_number(text)
    if not isinstance(v, float) or math.isnan(v):
        return v
    target = question_unit(question)
    if target is None:
        return v
    v2, span = parse_number_span(text)
    if span is None or v2 != v:
        return v
    m = _UNIT_AFTER_RE.match(text.strip(), span[1])
    if not m:
        return v
    src = _unit_scale(m.group(1))
    if src is None or src[0] != target[0]:
        return v
    return v * src[1] / target[1]


# ---------------------------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------------------------
def quantiphy_root():
    return os.path.join(LMUDataRoot(), 'QuantiPhy')


# Media integrity. Every file a key references is checked before the cache is accepted:
#  * the question table against a pinned sha256;
#  * every referenced HF video against its Git-LFS sha256 at the pinned revision. The per-file
#    (size, sha256) list comes from the HF API once, is cached next to the data, and is itself checked
#    against a pinned digest (so later loads need no network);
#  * hashes are cached per file with (size, mtime_ns); a normal load stat-checks, QUANTIPHY_VERIFY_MEDIA=1
#    re-hashes everything. A missing / empty / changed file is re-downloaded (hf_hub_download,
#    force_download) and re-checked; a second failure raises;
#  * QuantiPhy_test 480p transcodes are recorded (source sha256, ffmpeg settings version, own sha256,
#    frame count, size, mtime) and re-made on any mismatch;
#  * the TSV has a manifest (build version, TSV md5, digest of the (clip, sha256, frame count) set). On
#    reuse the TSV's own clip / n_frames columns are also compared with the verified media; any mismatch
#    rebuilds it.
SPLIT_FILES = {
    'validation': dict(repo=VAL_REPO, revision=VAL_REVISION, table='validation_dataset.csv',
                       table_sha256='4f29907c256371b998cf5f5b71287c12e2b3a60a87bacbb288036ffd35a1dc11',
                       manifest_sha256='53556d53f34064a6520dc97a22e6d4fe18550c7658003178dd0981b93763d168'),
    'test': dict(repo=TEST_REPO, revision=TEST_REVISION, table='test_dataset.parquet',
                 table_sha256='0eb2db32a22ca570ceeb1a6b6ff6d3f80405102934a75a17d1d38376bea7cca2',
                 manifest_sha256='dc9d9adf3775b27db1aa72ce46c6eb65b99c530e67029e066eead856e76c58af'),
}
BUILD_VERSION = 2       # TSV layout / content rules
TRANSCODE_VERSION = 1   # _transcode_480p settings
HF_MANIFEST = '.quantiphy_hf_manifest.json'
VERIFIED = '.quantiphy_verified.json'
TRANSCODES = '.quantiphy_480p_manifest.json'


def _verify_all():
    return os.environ.get('QUANTIPHY_VERIFY_MEDIA', '') == '1'


def _sha256(path, bufsize=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(bufsize), b''):
            h.update(buf)
    return h.hexdigest()


def _json_digest(obj):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _read_json(path, default=None):
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    import json
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _hf_video_manifest(src, info):
    """{rfilename: [size, sha256]} of every LFS mp4 at the pinned revision (cached, digest-pinned)."""
    path = os.path.join(src, HF_MANIFEST)
    m = _read_json(path)
    if m is None or _json_digest(m) != info['manifest_sha256']:
        from huggingface_hub import HfApi
        meta = HfApi().dataset_info(info['repo'], revision=info['revision'], files_metadata=True)
        m = {s.rfilename: [s.lfs.size, s.lfs.sha256] for s in meta.siblings
             if s.lfs and s.rfilename.endswith('.mp4')}
        if _json_digest(m) != info['manifest_sha256']:
            raise RuntimeError(f"{info['repo']}@{info['revision']}: LFS file list digest {_json_digest(m)} != pinned "
                               f"{info['manifest_sha256']}")
        os.makedirs(src, exist_ok=True)
        _write_json(path, m)
    return m


def _hf_download(info, src, rfilename):
    from huggingface_hub import hf_hub_download
    hf_hub_download(info['repo'], rfilename, repo_type='dataset', revision=info['revision'], local_dir=src,
                    force_download=True)


def _n_frames(path):
    import decord
    return len(decord.VideoReader(path))


def ensure_files(src, info, wanted, count_frames=False):
    """wanted: {rfilename: (size or None, sha256)}. Returns {rfilename: record} with sha256 (+ n_frames)."""
    state_path = os.path.join(src, VERIFIED)
    state = _read_json(state_path, {})
    verify_all = _verify_all()
    out, n_fixed = {}, 0
    try:
        for rf, (size, sha) in sorted(wanted.items()):
            p = os.path.join(src, rf)
            for attempt in range(2):
                st = os.stat(p) if os.path.isfile(p) else None
                rec = state.get(rf)
                good = False
                if st is not None and st.st_size > 0 and (size is None or st.st_size == size):
                    if (not verify_all and rec and rec.get('sha256') == sha and rec.get('size') == st.st_size
                            and rec.get('mtime_ns') == st.st_mtime_ns):
                        good = True
                    elif _sha256(p) == sha:
                        rec = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha)
                        good = True
                if good:
                    if count_frames and 'n_frames' not in rec:
                        rec['n_frames'] = _n_frames(p)
                    state[rf] = rec
                    break
                state.pop(rf, None)
                if attempt == 1:
                    raise RuntimeError(f'QuantiPhy: {p} missing or sha256 != {sha} after re-download')
                n_fixed += 1
                _hf_download(info, src, rf)
            out[rf] = state[rf]
    finally:
        _write_json(state_path, state)
    if n_fixed:
        print(f'[QuantiPhy] re-downloaded {n_fixed} missing/changed file(s) in {src}')
    return out


def _transcode_480p(src, dst):
    """Short side -> 480 (even other side), every frame kept at the source rate, no audio.
    The scale expression is evaluated by ffmpeg after its autorotation, so rotated phone clips also end
    with short side 480. (ffmpeg 4.4 here: -vsync passthrough; newer releases call it -fps_mode.)"""
    import decord
    vr = decord.VideoReader(src)
    scale = "scale='if(gte(iw,ih),-2,480)':'if(gte(iw,ih),480,-2)'"
    tmp = dst + '.tmp.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-vf', scale, '-vsync', 'passthrough', '-an',
           '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           tmp]
    subprocess.run(cmd, check=True)
    n_src, n_dst = len(vr), _n_frames(tmp)
    if n_src != n_dst:
        os.remove(tmp)
        raise RuntimeError(f'{src}: 480p transcode has {n_dst} frames, source {n_src}')
    os.replace(tmp, dst)
    return n_dst


def ensure_transcodes(src_records, src_dir, dst_dir):
    """src_records: {video_id: (source rfilename, source sha256)}. Returns {video_id: record}."""
    os.makedirs(dst_dir, exist_ok=True)
    man_path = os.path.join(dst_dir, TRANSCODES)
    man = _read_json(man_path, {})
    verify_all = _verify_all()
    n_made = 0
    try:
        for vid, (rf, src_sha) in sorted(src_records.items()):
            dst = os.path.join(dst_dir, f'{vid}.mp4')
            rec = man.get(vid)
            st = os.stat(dst) if os.path.isfile(dst) else None
            good = (st is not None and st.st_size > 0 and rec is not None and rec.get('src_sha256') == src_sha
                    and rec.get('version') == TRANSCODE_VERSION and rec.get('size') == st.st_size)
            if good and (verify_all or rec.get('mtime_ns') != st.st_mtime_ns):
                good = _sha256(dst) == rec.get('sha256')
                if good:
                    rec['mtime_ns'] = st.st_mtime_ns
            if not good:
                man.pop(vid, None)
                n = _transcode_480p(os.path.join(src_dir, rf), dst)
                st = os.stat(dst)
                man[vid] = dict(src_sha256=src_sha, version=TRANSCODE_VERSION, sha256=_sha256(dst), n_frames=n,
                                size=st.st_size, mtime_ns=st.st_mtime_ns)
                n_made += 1
                if n_made % 25 == 0:
                    _write_json(man_path, man)
    finally:
        _write_json(man_path, man)
    if n_made:
        print(f'[QuantiPhy] made {n_made} 480p transcode(s) in {dst_dir}')
    return {vid: man[vid] for vid in src_records}


def _file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(1 << 22), b''):
            h.update(buf)
    return h.hexdigest()


def tsv_is_current(tsv, manifest_path, media):
    """TSV reusable: manifest matches (version, md5, media digest) and its clip / n_frames columns
    agree with the verified media {clip: (sha256, n_frames)}."""
    man = _read_json(manifest_path)
    if not (os.path.exists(tsv) and man and man.get('build_version') == BUILD_VERSION
            and man.get('media_digest') == _json_digest(media) and man.get('tsv_md5') == _file_md5(tsv)):
        return False
    try:
        d = pd.read_csv(tsv, sep='\t', usecols=['video', 'n_frames'])
    except (ValueError, OSError):
        return False
    return all(v in media and int(n) == int(media[v][1]) for v, n in zip(d['video'], d['n_frames']))


# ---------------------------------------------------------------------------------------------
# Submission ids (official template) matched by question identity, never by row order
# ---------------------------------------------------------------------------------------------
_STABLE = ['video_id', 'video_source', 'inference_type', 'prior', 'fps']  # equal in template and parquet
_FUZZY_MIN_RATIO = 0.6     # question-text similarity a non-exact match needs
_FUZZY_MARGIN = 0.02       # best candidate must beat the runner-up by this much (both directions)


def _s(x):
    return '' if _isna(x) else str(x).strip()


def _identity(df, prior_col):
    """Per row: (stable key, full key). fps compared as float."""
    stable = list(zip(*[df[c].map(_s) if c != 'fps' else pd.to_numeric(df[c]).astype(float)
                        for c in ['video_id', 'video_source', 'inference_type']],
                      df[prior_col].map(_s), pd.to_numeric(df['fps']).astype(float)))
    full = [k + (_s(vt), _s(d), _s(q)) for k, vt, d, q in
            zip(stable, df['video_type'], df['depth_info'], df['question'])]
    return stable, full


def load_template(root):
    """Official submission template ($root/competition/), downloaded once, sha256-pinned."""
    import hashlib
    import urllib.request
    path = os.path.join(root, 'competition', 'quantiphy_submission_template.csv')

    def ok():
        return os.path.exists(path) and hashlib.sha256(open(path, 'rb').read()).hexdigest() == TEMPLATE_SHA256
    if not ok():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(TEMPLATE_URL, timeout=120) as r:
            data = r.read()
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
        if not ok():
            raise RuntimeError(f'{path}: sha256 != pinned {TEMPLATE_SHA256}')
    return pd.read_csv(path)


def match_template_ids(df, template, prior_col='prior'):
    """Template id for every row of df (test parquet / TSV), matched by question identity.

    1. exact: (video_id, video_source, inference_type, prior, fps, video_type, depth_info, question) equal;
    2. otherwise, within the same (video_id, video_source, inference_type, prior, fps) group, the
       mutually best question-text match (difflib ratio >= 0.6, ahead of the runner-up by >= 0.02 on both
       sides). The 2026-10 data fixes changed question spelling, depth_info or video_type in 99 rows.
    Raises ValueError on duplicate identities, ambiguous or missing matches. Returns (ids, modes)."""
    from difflib import SequenceMatcher
    t_stable, t_full = _identity(template, 'ground_truth_prior')
    d_stable, d_full = _identity(df, prior_col)
    t_ids = template['id'].astype(int).tolist()
    for name, keys in (('template', t_full), ('data', d_full)):
        dup = [k for k, c in pd.Series(keys).value_counts().items() if c > 1]
        if dup:
            raise ValueError(f'QuantiPhy {name}: {len(dup)} duplicated question identities, e.g. {dup[0]}')
    by_full = dict(zip(t_full, range(len(t_full))))
    ids, modes, used = [None] * len(df), [None] * len(df), set()
    for i, k in enumerate(d_full):
        j = by_full.get(k)
        if j is not None:
            ids[i], modes[i] = t_ids[j], 'exact'
            used.add(j)
    rest_d = [i for i in range(len(df)) if ids[i] is None]
    rest_t = [j for j in range(len(template)) if j not in used]
    for key in sorted({d_stable[i] for i in rest_d}, key=str):
        di = [i for i in rest_d if d_stable[i] == key]
        tj = [j for j in rest_t if t_stable[j] == key]
        while di:
            if not tj:
                raise ValueError(f'QuantiPhy: no template row left for {df.iloc[di[0]][["video_id", "question"]].tolist()}')
            sim = np.array([[SequenceMatcher(None, d_full[i][-1], t_full[j][-1]).ratio()
                             + 0.01 * (d_full[i][5] == t_full[j][5]) + 0.01 * (d_full[i][6] == t_full[j][6])
                             for j in tj] for i in di])
            progressed = False
            for a in range(len(di)):
                b = int(sim[a].argmax())
                if int(sim[:, b].argmax()) != a or sim[a, b] < _FUZZY_MIN_RATIO:
                    continue
                second_row = np.partition(sim[a], -2)[-2] if len(tj) > 1 else -1
                second_col = np.partition(sim[:, b], -2)[-2] if len(di) > 1 else -1
                if sim[a, b] - max(second_row, second_col) < _FUZZY_MARGIN:
                    continue
                i, j = di[a], tj[b]
                ids[i], modes[i] = t_ids[j], f'fuzzy:{sim[a, b]:.3f}'
                di.pop(a)
                tj.pop(b)
                progressed = True
                break
            if not progressed:
                raise ValueError(f'QuantiPhy: ambiguous template match for {len(di)} rows of {key[:3]}, e.g. '
                                 f'{df.iloc[di[0]]["question"]!r}')
    check_submission_ids(ids, t_ids)
    return ids, modes


def check_submission_ids(ids, template_ids):
    """Every template id exactly once (export guard)."""
    s = pd.Series(ids)
    if s.isna().any():
        raise ValueError(f'QuantiPhy submission: {int(s.isna().sum())} rows without a template id')
    dup = s[s.duplicated()].astype(int).unique().tolist()
    missing = sorted(set(int(x) for x in template_ids) - set(int(x) for x in s))
    extra = sorted(set(int(x) for x in s) - set(int(x) for x in template_ids))
    if dup or missing or extra:
        raise ValueError(f'QuantiPhy submission ids: {len(dup)} duplicated {dup[:5]}, {len(missing)} missing '
                         f'{missing[:5]}, {len(extra)} unknown {extra[:5]}')


class QuantiPhy(VideoBaseDataset):
    # 'Video-VQA' is not in CosmosReason.use_custom_prompt's exclusion list, but VIDEO datasets always
    # use their own build_prompt in inference_video.py; the type only matters for DATASET_TYPE lookups.
    TYPE = 'Video-VQA'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='QuantiPhy', pack=False, nframe=0, fps=-1, **kwargs):
        if dataset not in SUBSETS:
            raise ValueError(f'Unknown QuantiPhy key {dataset}; choose from {list(SUBSETS)}')
        if (nframe or 0) > 0 or (fps is not None and fps > 0):
            raise ValueError(f'{dataset}: the official protocol sends every frame; nframe/fps must not be set')
        self.split = SUBSETS[dataset]
        super().__init__(dataset=dataset, pack=pack, nframe=0, fps=-1)

    @classmethod
    def supported_datasets(cls):
        return list(SUBSETS)

    # -- data -------------------------------------------------------------------------------
    def prepare_dataset(self, dataset):
        root = quantiphy_root()
        os.makedirs(root, exist_ok=True)
        tsv = os.path.join(LMUDataRoot(), f'{dataset}.tsv')
        man_path = os.path.join(LMUDataRoot(), f'{dataset}.manifest.json')
        info = SPLIT_FILES[self.split]
        src = os.path.join(root, self.split)
        with portalocker.Lock(os.path.join(root, '.prepare.lock'), 'w', timeout=7200):
            ensure_files(src, info, {info['table']: (None, info['table_sha256'])})
            df = self._read_table(src)
            hfm = _hf_video_manifest(src, info)
            by_id = {}
            for rf in hfm:  # basename stripped: the validation repo has 'validation_videos/ captured_0041x.mp4'
                if self.split == 'validation' and not rf.startswith('validation_videos/'):
                    continue
                by_id[os.path.basename(rf).strip()[:-4]] = rf
            vids = sorted(set(df['video_id']))
            missing = [v for v in vids if v not in by_id]
            if missing:
                raise RuntimeError(f'QuantiPhy {self.split}: {len(missing)} videos not in the HF repo, e.g. {missing[:3]}')
            recs = ensure_files(src, info, {by_id[v]: tuple(hfm[by_id[v]]) for v in vids},
                                count_frames=self.split == 'validation')
            if self.split == 'validation':
                video_root = src
                clip_of = {v: by_id[v] for v in vids}
                media = {by_id[v]: [recs[by_id[v]]['sha256'], recs[by_id[v]]['n_frames']] for v in vids}
            else:
                video_root = os.path.join(root, 'test_480p')
                tr = ensure_transcodes({v: (by_id[v], recs[by_id[v]]['sha256']) for v in vids}, src, video_root)
                clip_of = {v: f'{v}.mp4' for v in vids}
                media = {f'{v}.mp4': [tr[v]['sha256'], tr[v]['n_frames']] for v in vids}
            if not tsv_is_current(tsv, man_path, media):
                out = self._build_tsv(root, df, clip_of, media)
                tmp = tsv + '.tmp.tsv'
                out.to_csv(tmp, sep='\t', index=False)
                os.replace(tmp, tsv)
                _write_json(man_path, dict(build_version=BUILD_VERSION, split=self.split, repo=info['repo'],
                                           revision=info['revision'], table_sha256=info['table_sha256'],
                                           tsv_md5=_file_md5(tsv), media_digest=_json_digest(media),
                                           n_rows=len(out), n_videos=len(media)))
        return dict(root=video_root, data_file=tsv)

    def _read_table(self, src):
        if self.split == 'validation':
            return pd.read_csv(os.path.join(src, 'validation_dataset.csv'), index_col=0)
        return pd.read_parquet(os.path.join(src, 'test_dataset.parquet'))

    def _build_tsv(self, root, df, clip_of, media):
        if self.split == 'validation':
            ids = df.index.astype(int)
            id_match = ['official'] * len(df)
            gt = df['ground_truth_posterior'].astype(float)
            prior = df['ground_truth_prior']
        else:
            ids, id_match = match_template_ids(df, load_template(root))  # by question identity, not row order
            gt = pd.Series([np.nan] * len(df))
            prior = df['prior']
        nfr = {v: media[clip_of[v]][1] for v in clip_of}
        vidx = clip_of
        return pd.DataFrame({
            'index': ids,
            'video_id': df['video_id'].values,
            'video_source': df['video_source'].values,
            'video_type': df['video_type'].values,
            'fps': df['fps'].values,
            'inference_type': df['inference_type'].values,
            'question': df['question'].values,
            'prior': prior.values,
            'depth_info': df['depth_info'].values,
            'answer': gt.values,
            'video': [vidx[v] for v in df['video_id']],
            'n_frames': [nfr[v] for v in df['video_id']],
            'id_match': id_match,
        })

    def video_path(self, video):
        return os.path.join(self.data_root, video)

    def save_video_frames(self, video, n_frames=None):
        """Every frame as JPEG (non-VIDEO_LLM path; the official request sends all frames)."""
        import decord
        from PIL import Image

        stem = os.path.splitext(os.path.basename(video).strip())[0]
        frame_dir = os.path.join(self.frame_root, stem)
        os.makedirs(frame_dir, exist_ok=True)
        lock_path = os.path.join(self.frame_root, stem + '.lock')
        with portalocker.Lock(lock_path, 'w', timeout=600):
            vr = decord.VideoReader(self.video_path(video))
            n = len(vr)
            paths = [os.path.join(frame_dir, f'frame-{i + 1}-of-{n}.jpg') for i in range(n)]
            for i, p in enumerate(paths):
                if not os.path.exists(p):
                    tmp = p + '.tmp.jpg'
                    Image.fromarray(vr[i].asnumpy()).save(tmp, quality=95)
                    os.replace(tmp, p)
        return paths

    @staticmethod
    def video_nframes(n_frames):
        """Frames requested from qwen-vl-utils: the largest even count <= n_frames."""
        return int(n_frames) - int(n_frames) % 2

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        text = build_user_text(line['question'], line['prior'], line['depth_info'], line['video_type'])
        # role='system' -> BaseAPI.preprocess_message_with_role moves it into the system prompt.
        msgs = [dict(role='system', type='text', value=SYSTEM_PROMPT)]
        if video_llm:
            msgs.append(dict(type='video', value=self.video_path(line['video']),
                             nframes=self.video_nframes(line['n_frames'])))
        else:
            msgs.extend(dict(type='image', value=p) for p in self.save_video_frames(line['video']))
        msgs.append(dict(type='text', value=text))
        return msgs

    # -- scoring ----------------------------------------------------------------------------
    @staticmethod
    def official_frame(data):
        """Prediction table -> official evaluator.py input (first column = id, parsed_value)."""
        return pd.DataFrame({
            'id': data['index'].astype(int).values,
            'video_id': data['video_id'].values,
            'video_source': data['video_source'].values,
            'video_type': data['video_type'].values,
            'fps': data['fps'].values,
            'inference_type': data['inference_type'].values,
            'question': data['question'].values,
            'ground_truth_prior': data['prior'].values,
            'depth_info': data['depth_info'].values,
            'ground_truth_posterior': pd.to_numeric(data['answer'], errors='coerce').values,
            'parsed_value': [parse_number(p) for p in data['prediction']],
        })

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data['prediction'] = [None if _isna(p) else str(p) for p in data['prediction']]
        off = self.official_frame(data)
        if self.split == 'test':
            # ids come from match_template_ids (TSV build); re-check against the template before export
            template = load_template(quantiphy_root())
            check_submission_ids(off['id'].tolist(), template['id'].tolist())
            fresh, _ = match_template_ids(data, template)  # re-derive from the rows' own content
            bad = [(int(a), int(b)) for a, b in zip(off['id'], fresh) if int(a) != int(b)]
            if bad:
                raise ValueError(f'QuantiPhy submission: {len(bad)} rows carry an id that does not match their '
                                 f'question in the template (stored, expected) e.g. {bad[:3]}')
            sub_file = get_intermediate_file_path(eval_file, '_quantiphy_submission', 'csv')
            off.sort_values('id')[SUBMISSION_COLUMNS].to_csv(sub_file, index=False)
            res = OrderedDict(n=len(off), invalid_percentage=invalid_count(off['parsed_value']) / len(off) * 100)
            print(f'[{self.dataset_name}] GT withheld; submission CSV written to {sub_file}: {dict(res)}')
            return res

        off_file = get_intermediate_file_path(eval_file, '_quantiphy_official', 'csv')
        off.to_csv(off_file, index=False)  # runnable with the official evaluator.py
        scored, summ = score_frame(off, gt_col='ground_truth_posterior')
        off['parsed_value_lenient'] = [lenient_value(p, q) for p, q in zip(data['prediction'], data['question'])]
        scored_u, summ_u = score_frame(off, parsed_col='parsed_value_lenient', gt_col='ground_truth_posterior',
                                       mra_col='mra_lenient')
        scored['parsed_value_lenient'] = off['parsed_value_lenient'].values
        scored['mra_lenient'] = scored_u['mra_lenient'].values
        scored['prediction'] = data['prediction'].values
        dump(scored, get_intermediate_file_path(eval_file, '_quantiphy_score', 'xlsx'))

        def pct(x):
            return None if x is None or (isinstance(x, float) and math.isnan(x)) else 100.0 * x

        res = OrderedDict()
        res['Overall'] = pct(summ['mra_average'])
        for c in CATEGORIES:
            res[c] = pct(summ[f'mra_{c}'])
        for k, v in summ.items():
            if k.startswith('mra_bg_') or k.startswith('mra_obj_'):
                res[k[4:]] = pct(v)
        res['invalid_percentage'] = summ['invalid_percentage']
        res['n'] = len(off)
        res['n_unparsed'] = int(pd.isna(off['parsed_value']).sum())
        res['Overall_lenient'] = pct(summ_u['mra_average'])
        for c in CATEGORIES:
            res[f'{c}_lenient'] = pct(summ_u[f'mra_{c}'])
        res['n_lenient_changed'] = int(((off['parsed_value_lenient'] != off['parsed_value'])
                                       & off['parsed_value'].notna()).sum())
        table = pd.DataFrame([dict(metric=k, value=v) for k, v in res.items()])
        table.to_csv(get_intermediate_file_path(eval_file, '_acc', 'csv'), index=False)
        print(f'[{self.dataset_name}] (official: {OFFICIAL_CODE})\n'
              + table.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        return res
