"""3DSRBench with the official headline metric (CircularEval; flipped images scored as separate questions).

Paper: arXiv 2412.07825 (v4), https://3dsrbench.github.io/   Data: HF ccvl/3DSRBench @ 5c90e422.

The upstream class `_3DSRBench` (image_mcq.py, key `3DSRBench`) evaluates all 11,686 TSV rows and
returns a 4-row DataFrame (vanilla / flip_eval / circ_eval / flip_circ_eval). The paper's numbers
and the authors' scorer `compute_3drbench_results_circular.py` (HF ccvl/3DSRBench @
5c90e422b68d0484d50a3ff06b3096ea7d288026) use CircularEval only: a question counts as correct iff
every option ordering (qid suffix "", -1, -2, -3) is answered correctly; "<id>-flip" copies are
separate questions (no flip pairing). Overall / Height / Location / Orientation / Multi-object are
means over questions (not over sub-types). This equals upstream's `circ_eval` row, but parse_score
does not pick that row as the headline (see PROTOCOL.md), so this key returns a flat dict with
`Overall` = official CircularEval and the other settings as diagnostics. The scored rows are validated
against the TSV's copy structure first (score_validated): a question with a missing ordering or a
duplicated copy is excluded from Overall and flagged, and evaluation_scope is 'full' only for the
complete release (the official script would multiply whatever copies are present).

Same TSV (OpenCompass 3DSRBench.tsv, md5 610516a0b4710595545b7613c60524e8), same prompt
(ImageMCQDataset default, as the official README runs `run.py --data 3DSRBenchv1`), same answer
extraction (mcq_vanilla_eval) as the upstream key; only the aggregation differs.
"""

import os
import os.path as osp
import warnings

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, load, md5
from vlmeval.smp.file import get_intermediate_file_path
from .image_mcq import _3DSRBench

# Mapping verbatim from compute_3drbench_results_circular.py.
MAPPING = {
    'location': ['location_above', 'location_closer_to_camera', 'location_next_to'],
    'height': ['height_higher'],
    'orientation': ['orientation_in_front_of', 'orientation_on_the_left', 'orientation_viewpoint'],
    'multi_object': ['multi_object_closer_to', 'multi_object_facing', 'multi_object_viewpoint_towards_object',
                     'multi_object_parallel', 'multi_object_same_direction']}
TYPES = ['height', 'location', 'orientation', 'multi_object']
SUBTYPES = sum([MAPPING[k] for k in TYPES], [])

TSV_MD5 = '610516a0b4710595545b7613c60524e8'
# The openxlab.space host serves this file from the Aliyun OSS bucket below; its TLS certificate
# expired (2026-11), the bucket endpoint has a valid one. Same file (md5 checked).
TSV_URLS = ['https://opencompass.openxlab.space/utils/VLMEval/3DSRBench.tsv',
            'https://opencompass.oss-cn-shanghai.aliyuncs.com/utils/VLMEval/3DSRBench.tsv']
N_ROWS = 11686
N_QUESTIONS = 5157   # 2,625 base + 2,532 flipped questions after collapsing option orderings


def official_group_key(qid: str) -> str:
    """compute_3drbench_results_circular.py: strip a trailing '-N' ordering suffix."""
    return qid[:-2] if qid[-2] == '-' else qid


def official_circular(qids, categories, hits) -> dict:
    """Port of compute_3drbench_results_circular.py (values 0-100, unrounded)."""
    results = {}
    for qid, cat, hit in zip(qids, categories, hits):
        assert hit in [0, 1], hit
        assert cat in SUBTYPES, cat
        k = official_group_key(str(qid))
        if k in results:
            results[k][0] = results[k][0] * hit
        else:
            results[k] = [hit, cat]
    out = {'overall': float(np.mean([results[k][0] for k in results])) * 100}
    for t in TYPES:
        vals = [results[k][0] for k in results if results[k][1] in MAPPING[t]]
        if vals:
            out[t] = float(np.mean(vals)) * 100
    for t in SUBTYPES:
        vals = [results[k][0] for k in results if results[k][1] == t]
        if vals:
            out[t] = float(np.mean(vals)) * 100
    out['n_questions'] = len(results)
    return out


def validate_copies(qids, meta_qids) -> dict:
    """Check the scored rows against the TSV's copy structure.

    Expected copies of a question = every TSV qid with the same official group key (all option orderings).
    Returns {complete: group keys whose rows are exactly the expected copies, each once; incomplete: group
    keys with a missing ordering; duplicated: group keys with a qid scored more than once;
    n_duplicate_rows; n_missing_questions (TSV questions with no row at all)}.
    """
    expected = {}
    for q in meta_qids:
        expected.setdefault(official_group_key(str(q)), set()).add(str(q))
    seen, counts = {}, {}
    for q in qids:
        q = str(q)
        counts[q] = counts.get(q, 0) + 1
        seen.setdefault(official_group_key(q), set()).add(q)
    duplicated = {official_group_key(q) for q, c in counts.items() if c > 1}
    unknown = {k for k in seen if k not in expected}
    if unknown:
        raise ValueError(f'qids not in the 3DSRBench TSV, e.g. {sorted(unknown)[:3]}')
    incomplete = {k for k, v in seen.items() if v != expected[k]}
    complete = set(seen) - incomplete - duplicated
    return dict(complete=complete, incomplete=incomplete, duplicated=duplicated,
                n_duplicate_rows=sum(c - 1 for c in counts.values()),
                n_missing_questions=len(set(expected) - set(seen)), n_release_questions=len(expected))


def score_validated(qids, categories, hits, meta_qids) -> dict:
    """Official CircularEval over the questions whose copies are exactly the TSV's (each once).

    Questions with a missing ordering or a duplicated copy are EXCLUDED from Overall and flagged (the
    official script multiplies whatever copies are present, so one correct copy with the others missing
    would count as correct). `overall_present_copies` = the official script on all rows as given.
    evaluation_scope is 'full' only if every release question is scored with all its copies exactly once.
    """
    v = validate_copies(qids, meta_qids)
    keep = [official_group_key(str(q)) in v['complete'] for q in qids]
    rows = [(q, c, h) for q, c, h, k in zip(qids, categories, hits, keep) if k]
    out = official_circular(*zip(*rows)) if rows else {'overall': 0.0, 'n_questions': 0}
    full = (not v['incomplete'] and not v['duplicated'] and v['n_missing_questions'] == 0
            and out['n_questions'] == N_QUESTIONS)
    out.update(
        evaluation_scope='full' if full else 'subset', headline_partial=int(not full),
        n_incomplete_excluded=len(v['incomplete'] - v['duplicated']),
        n_duplicated_excluded=len(v['duplicated']), n_duplicate_rows=v['n_duplicate_rows'],
        n_missing_questions=v['n_missing_questions'],
        overall_present_copies=official_circular(qids, categories, hits)['overall'] if len(qids) else 0.0)
    return out


def _grouped_overall(qids, hits, keyfn) -> float:
    acc = {}
    for q, h in zip(qids, hits):
        k = keyfn(str(q))
        acc[k] = acc.get(k, 1) * h
    return float(np.mean(list(acc.values()))) * 100


class ThreeDSRBenchOfficial(_3DSRBench):
    DATASET_URL = {'3DSRBench_official': TSV_URLS[0]}
    DATASET_MD5 = {'3DSRBench_official': TSV_MD5}

    @classmethod
    def supported_datasets(cls):
        return ['3DSRBench_official']

    def __init__(self, dataset='3DSRBench_official', **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        # Share decoded images with the upstream key (same TSV, same index-based file names).
        self.img_root = osp.join(LMUDataRoot(), 'images', '3DSRBench')

    def load_data(self, dataset):
        """Reuse $LMUData/3DSRBench.tsv (the upstream key's file); download it if missing / md5 differs."""
        from vlmeval.smp import download_file
        path = osp.join(LMUDataRoot(), '3DSRBench.tsv')
        self.data_path = path
        if not (osp.exists(path) and md5(path) == TSV_MD5):
            for url in TSV_URLS:
                try:
                    tmp = f'{path}.{os.getpid()}.part'
                    download_file(url, tmp)
                    if md5(tmp) == TSV_MD5:
                        os.replace(tmp, path)
                        break
                    os.remove(tmp)
                except Exception as e:   # noqa: BLE001
                    warnings.warn(f'3DSRBench download from {url} failed: {e}')
            else:
                raise RuntimeError(f'could not obtain 3DSRBench.tsv with md5 {TSV_MD5}')
        return load(path)

    def evaluate(self, eval_file, **judge_kwargs):
        # Hits exactly as upstream _3DSRBench.evaluate (same judge handling, mcq_vanilla_eval, result file
        # names); upstream's own grouping is not called because it raises KeyError on partial inputs
        # (e.g. '<id>-flip-1' present but '<id>-1' missing). Its four settings are recomputed below.
        import string

        from vlmeval.dataset.utils import DEBUG_MESSAGE, build_judge
        from vlmeval.dataset.utils.multiple_choice import mcq_vanilla_eval
        nproc = judge_kwargs.pop('nproc', 4)
        model = judge_kwargs.get('model', 'exact_matching')
        name_str = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}.get(model, model)
        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')
        data = load(eval_file).sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for k in list(data.keys()):
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)
        meta = self.data
        assert set(data['index']) <= set(meta['index']), \
            f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        dump(data, get_intermediate_file_path(eval_file, '_result', 'xlsx'))
        hits = [int(h) for h in data['hit']]
        off = score_validated(list(data['qid']), list(data['category']), hits, list(self.data['qid']))

        summary = {'Overall': off['overall']}
        for t in TYPES + SUBTYPES:
            if t in off:
                summary[t] = off[t]
        for k in ('n_questions', 'evaluation_scope', 'headline_partial', 'n_incomplete_excluded',
                  'n_duplicated_excluded', 'n_duplicate_rows', 'n_missing_questions'):
            summary[k] = off[k]
        summary['Overall_present_copies'] = off['overall_present_copies']
        if off['headline_partial']:
            warnings.warn(f'[{self.dataset_name}] partial input: Overall covers {off["n_questions"]} of '
                          f'{N_QUESTIONS} questions ({off["n_incomplete_excluded"]} with missing orderings and '
                          f'{off["n_duplicated_excluded"]} with duplicated copies excluded)')
        # Diagnostics (not the paper metric): upstream's other settings.
        summary['vanilla'] = float(np.mean(hits)) * 100
        summary['flip_eval'] = _grouped_overall(data['qid'], hits, lambda x: x.replace('-flip', ''))
        summary['flip_circ_eval'] = _grouped_overall(data['qid'], hits, lambda x: x[:8])
        # upstream's circ_eval grouping key (x[:8], or x[:13] for flips), over all rows as given
        summary['upstream_circ_eval'] = _grouped_overall(
            data['qid'], hits, lambda x: x[:8] if '-flip' not in x else x[:13])

        dump(pd.DataFrame([summary]), get_intermediate_file_path(eval_file, '_official_acc', 'csv'))
        print(f'[{self.dataset_name}] official CircularEval: {summary}')
        return summary
