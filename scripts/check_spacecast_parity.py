#!/usr/bin/env python
"""SpaceCast-Bench parity check: official evaluator vs vlmeval SpaceCastBench (no server, no GPU).

Loads the official `spacecast_eval` package at the pinned commit (downloaded once into
$LMUData/SpaceCast-Bench/official_<commit>/) and the HF parquet, then checks

  1. prompts: official build_prompt(row, mode) == the text our build_prompt sends, both modes, every
     question; our message = [system prompt, user text, images...] (official api_messages order);
  2. images: same order as the parquet `images`, and (for --image-sample questions) our JPEG file
     bytes == official prepare_image(blob) bytes;
  3. scoring: a fixed set of synthetic responses (letters, "Answer: X", CoT text, option text,
     multi-select lists, garbage, empty, API failure) is scored by official scoring.score and by
     SpaceCastBench.evaluate -> identical per-question correctness and identical macro / per-category
     accuracy; GT-as-prediction -> 100.

    LMUData=/data0/Cosmos3-Reasoner/LMUData python scripts/check_spacecast_parity.py
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OFFICIAL_COMMIT = '6696e5eb41371a9c8038c0d902d2aa488b800593'
OFFICIAL_FILES = ['__init__.py', 'protocol.py', 'scoring.py', 'data.py', 'loading.py', 'backends.py']
RAW = 'https://raw.githubusercontent.com/ZJU-REAL/SpaceCast-Bench/{commit}/spacecast_eval/{name}'
FAIL_MSG = 'Failed to obtain answer via API.'


def fetch_official(root: Path) -> Path:
    base = root / 'SpaceCast-Bench' / f'official_{OFFICIAL_COMMIT[:12]}'
    pkg = base / 'spacecast_eval'
    pkg.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_FILES:
        dst = pkg / name
        if not dst.exists():
            with urllib.request.urlopen(RAW.format(commit=OFFICIAL_COMMIT, name=name), timeout=60) as r:
                dst.write_bytes(r.read())
    return base


def synthetic_response(i: int, row: dict) -> str | None:
    """Deterministic mix of right/wrong, well-formed/odd responses for question i."""
    n = len(row['options'])
    gold = sorted(row['answer'])
    letters = [chr(65 + k) for k in range(n)]
    wrong = next(x for x in letters if x not in gold)
    g = gold[0]
    opts = row['options']
    t = [
        g,                                                    # bare correct letter
        wrong,                                                # bare wrong letter
        f'Answer: {g}',
        f'Reasoning: the chair moves left of the table.\nAnswer: {wrong}',
        f'**Answer: ({g})**',
        f'{g}) {opts[ord(g) - 65]}',
        f'The answer is {g}.',
        f'Answer: {opts[ord(g) - 65]}',                       # option text
        f'So {opts[ord(wrong) - 65]}',
        'I cannot tell from these views.',                    # unparsable
        '',                                                   # empty
        None,                                                 # missing / API failure
        FAIL_MSG,
        f'Answer: {g}\nMaybe {wrong}',
        f'{g.lower()}',
        f'Answer: {", ".join(gold)}',                         # multi-select formatted list
        f'Answer: {" and ".join(gold)}',
        ','.join(letters[:2]),
        f'...makes sense.{g}',
        f'<think>hmm</think>\n\nAnswer: {g}',
    ]
    return t[i % len(t)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image-sample', type=int, default=200, help='questions whose image bytes are compared')
    args = ap.parse_args(argv)

    root = Path(os.environ.get('LMUData', '/data0/Cosmos3-Reasoner/LMUData'))
    os.environ['LMUData'] = str(root)
    warnings.filterwarnings('ignore')
    sys.path.insert(0, str(fetch_official(root)))
    O_protocol = importlib.import_module('spacecast_eval.protocol')
    O_scoring = importlib.import_module('spacecast_eval.scoring')
    O_backends = importlib.import_module('spacecast_eval.backends')
    sys.path.insert(0, str(REPO / 'third_party' / 'vlmevalkit-cosmos'))
    from vlmeval.dataset import SpaceCastBench
    from vlmeval.dataset.spacecast import HF_FILE
    from vlmeval.smp import dump
    import pyarrow.parquet as pq

    parquet = root / 'SpaceCast-Bench' / 'hf' / Path(HF_FILE).name
    ds_direct = SpaceCastBench('SpaceCastBench')          # builds TSV + images on first use
    ds_cot = SpaceCastBench('SpaceCastBench_CoT')
    table = pq.read_table(parquet, columns=None)
    gold_cols = [c for c in table.column_names if c != 'images']
    gold = table.select(gold_cols).to_pylist()
    data = ds_direct.data
    assert len(gold) == len(data), (len(gold), len(data))
    assert [r['question_uid'] for r in gold] == list(data['question_uid']), 'question order differs'
    errors: list[str] = []

    # 1. prompts + message structure
    n_img = []
    for i, row in enumerate(gold):
        for ds, mode in ((ds_direct, 'direct'), (ds_cot, 'cot')):
            msgs = ds.build_prompt(data.iloc[i])
            if not (msgs[0].get('role') == 'system' and msgs[0]['value'] == O_protocol.SYSTEM_PROMPT):
                errors.append(f'{row["question_uid"]} {mode}: system prompt')
            if msgs[1]['type'] != 'text' or msgs[1]['value'] != O_protocol.build_prompt(row, mode):
                errors.append(f'{row["question_uid"]} {mode}: prompt text differs')
            imgs = [m['value'] for m in msgs[2:]]
            if any(m['type'] != 'image' for m in msgs[2:]):
                errors.append(f'{row["question_uid"]} {mode}: non-image after the text')
            want = [os.path.join(ds.img_root, p if p.lower().endswith(('.jpg', '.jpeg')) else p + '.jpg')
                    for p in row['image_paths']]
            if imgs != want:
                errors.append(f'{row["question_uid"]} {mode}: image order/paths differ')
        n_img.append(len(row['image_paths']))
    api = O_backends.api_messages('P', [])
    assert [m['role'] for m in api] == ['system', 'user'] and api[1]['content'][0]['type'] == 'text'
    print(f'[prompts] {len(gold)} questions x 2 modes checked; images/question '
          f'min {min(n_img)} max {max(n_img)} mean {sum(n_img) / len(n_img):.2f}')

    # 2. image bytes for a sample (official prepare_image on the parquet bytes)
    step = max(1, len(gold) // max(1, args.image_sample))
    sample = set(range(0, len(gold), step))
    checked = 0
    for b in pq.ParquetFile(parquet).iter_batches(batch_size=32, columns=['question_uid', 'images', 'image_paths']):
        for row in b.to_pylist():
            i = data.index[data['question_uid'] == row['question_uid']][0]
            if i not in sample:
                continue
            for item, rel in zip(row['images'], row['image_paths']):
                rel = rel if rel.lower().endswith(('.jpg', '.jpeg')) else rel + '.jpg'
                ours = Path(ds_direct.img_root, rel).read_bytes()
                if ours != O_protocol.prepare_image(item['bytes']):
                    errors.append(f'{row["question_uid"]}: image bytes differ for {rel}')
                checked += 1
    print(f'[images] {checked} images from {len(sample)} questions byte-compared with official prepare_image')

    # 3. scoring parity
    def run_both(responses, tag):
        preds = []
        for row, resp in zip(gold, responses):
            if resp is None or resp == FAIL_MSG:
                # as written by official run.py when a call fails after its retries
                preds.append({'question_uid': row['question_uid'], 'raw_response': None, 'prediction': None,
                              'status': 'error', 'error_type': 'APIError'})
            else:
                preds.append({'question_uid': row['question_uid'], 'raw_response': resp, 'prediction': None})
        o_report, o_details = O_scoring.score(gold, preds)
        o_correct = {d['question_uid']: d['correct'] for d in o_details}
        with tempfile.TemporaryDirectory() as td:
            df = data.copy()
            df['prediction'] = [r if r is not None else float('nan') for r in responses]
            ef = os.path.join(td, f'parity_SpaceCastBench_{tag}.xlsx')
            dump(df, ef)
            ours = ds_direct.evaluate(ef)
            import pandas as pd
            scored = pd.read_excel(os.path.join(td, f'parity_SpaceCastBench_{tag}_score.xlsx'))
        mism = [u for u, h in zip(scored['question_uid'], scored['hit']) if bool(h) != o_correct[u]]
        if mism:
            errors.append(f'[{tag}] {len(mism)} per-question mismatches, e.g. {mism[:5]}')
        o_macro = 100 * o_report['macro_type_accuracy']
        if not math.isclose(ours['Overall'], o_macro, abs_tol=1e-9):
            errors.append(f'[{tag}] macro {ours["Overall"]} != official {o_macro}')
        for k, v in o_report['by_type'].items():
            if not math.isclose(ours[k], 100 * v['accuracy'], abs_tol=1e-9):
                errors.append(f'[{tag}] {k}: {ours[k]} != {100 * v["accuracy"]}')
        if not math.isclose(ours['micro_accuracy'], 100 * o_report['overall']['accuracy'], abs_tol=1e-9):
            errors.append(f'[{tag}] micro differs')
        print(f'[score:{tag}] official macro {o_macro:.4f} / ours {ours["Overall"]:.4f}; '
              f'per-question mismatches {len(mism)}/{len(gold)}; '
              f'L1/L2/L3 macro {ours["L1_macro"]:.2f}/{ours["L2_macro"]:.2f}/{ours["L3_macro"]:.2f}; '
              f'scope {ours["evaluation_scope"]}')
        return ours

    synth = [synthetic_response(i, r) for i, r in enumerate(gold)]
    run_both(synth, 'synthetic')
    gt = run_both([','.join(sorted(r['answer'])) for r in gold], 'gt')
    if not math.isclose(gt['Overall'], 100.0):
        errors.append(f'GT-as-prediction gives {gt["Overall"]}')
    counts = data['category'].value_counts().to_dict()
    print('[counts] ' + json.dumps(dict(sorted(counts.items()))))

    if errors:
        print(f'FAIL: {len(errors)} problems')
        for e in errors[:30]:
            print('  ' + e)
        return 1
    print('OK: prompts, image order/bytes and scoring match the official evaluator')
    return 0


if __name__ == '__main__':
    sys.exit(main())
