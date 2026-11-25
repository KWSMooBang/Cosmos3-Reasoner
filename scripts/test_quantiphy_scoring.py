#!/usr/bin/env python
"""Offline tests of the QuantiPhy port against the official code (no GPU / server).

Official sources (github.com/Paulineli/QuantiPhy @ aa8daa7, MIT) are fetched once into
$LMUData/QuantiPhy/official/<commit>/ and sha256-checked:
  * run_API_results.py: parse_number / num_re / EXACT_NUMBER_WITH_UNIT_RE / build_context_prefix are
    taken from its AST and exec'd (the module itself is never imported: it pip-installs at import time);
  * evaluator.py: run unmodified as a subprocess (python -I) on the CSV our evaluate() exports;
  * quantiphy_validation.csv (GT, == HF validation_dataset.csv) and model_outputs/gpt-5.1.csv +
    mra_results/all_model_results.csv (the official GPT-5.1 run and its published scores).

Tests
  1. prompt parity: our build_user_text / build_context_prefix == official on every validation row.
  2. parse parity: our parse_number == official parse_number on a synthetic response set (all rows x
     templates: bare numbers, units, delimiters, sci notation, negatives, prose, empty, junk, ...).
  3. scorer parity: QuantiPhy.evaluate on synthetic predictions -> exported CSV -> official evaluator.py;
     every official metric (4 categories, average, bg/obj breakdowns, invalid_percentage) == ours.
  4. published numbers: our score_frame on the official gpt-5.1.csv == mra_results/all_model_results.csv.
  5. GT as prediction ("<gt> <unit>") -> 100; garbage (no number / random numbers) -> low.
  6. unit/number parse cases with expected official values; diagnostic lenient cases (exponent fix + unit conversion).

    python scripts/test_quantiphy_scoring.py
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

COMMIT = 'aa8daa7a5736ce3514ed985189ad27ef4a2e6e68'
FILES = {
    'model_run_example/run_API_results.py': '53f5d8c798e50cfba592a76548d3fa5f5e6b14a864f21ebb170a629caf34e05b',
    'evaluator.py': '98c741f52d69ebd5479d27c392e34545e593f00b67f9ab305908b148a46f9913',
    'quantiphy_validation.csv': '4f29907c256371b998cf5f5b71287c12e2b3a60a87bacbb288036ffd35a1dc11',
    'model_outputs/gpt-5.1.csv': None,
    'mra_results/all_model_results.csv': None,
}

FAILS: list[str] = []


def check(ok, msg):
    print(f"[{'OK' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAILS.append(msg)


def fetch_official(root: Path) -> Path:
    d = root / 'official' / COMMIT
    for rel, sha in FILES.items():
        p = d / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://raw.githubusercontent.com/Paulineli/QuantiPhy/{COMMIT}/{rel}'
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            tmp = p.with_suffix(p.suffix + '.tmp')
            tmp.write_bytes(data)
            os.replace(tmp, p)
        if sha:
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != sha:
                raise SystemExit(f'{p}: sha256 {got} != {sha}')
    return d


def official_namespace(run_api: Path) -> dict:
    """num_re, EXACT_NUMBER_WITH_UNIT_RE, parse_number, build_context_prefix from the official script."""
    import re
    tree = ast.parse(run_api.read_text())
    want = {'num_re', 'EXACT_NUMBER_WITH_UNIT_RE', 'parse_number', 'build_context_prefix'}
    nodes = [n for n in tree.body
             if (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in want for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name in want)]
    ns = {'re': re, 'np': np, 'pd': pd}
    exec(compile(ast.Module(nodes, []), str(run_api), 'exec'), ns)
    assert want <= set(ns), want - set(ns)
    return ns


def official_user_text(ns, row):
    """Official zero_shot_prompt text for version 3 (include_depth for video_type[1] == '3')."""
    vt = str(row.get('video_type', ''))
    depth = row.get('depth_info') if (len(vt) >= 2 and vt[1] == '3') else None
    return f"{ns['build_context_prefix'](row['ground_truth_prior'], depth)}{row['question']}" \
           "\n\nPlease answer the question with numbers and units ONLY. No explanation needed."


UNITS = ['m', 'cm', 'mm', 'meters', 'm/s', 'cm/s', 'm/s^2', 'm/s²', 'cm/s2', 'km/h', 'ft', 'inches', 'g', 's']


def synthetic_responses(gt: float, rng: np.random.Generator) -> list[str]:
    u = UNITS[rng.integers(len(UNITS))]
    x = float(gt) * float(rng.choice([0.1, 0.5, 0.8, 0.95, 1.0, 1.05, 1.3, 2.0, 10.0]))
    return [
        f'{x:.3g} {u}', f'{x:.4f}{u}', f'{x}', f'-{x:.2f} {u}', f'{x:.2e} {u}', f'Answer: {x:.2f} {u}',
        f'Final Answer: {x:.1f} {u}.', f'The speed is {x:.2f} {u}', f'v = {x:.3f} {u}', f'=> {x:.2f}',
        f'Approximately {x:.1f}-{x * 1.2:.1f} {u}', f'{x:,.2f} {u}', f'At t=2s: {x:.2f} {u}',
        f'I estimate {x:.2f} {u} (uncertain)', f'{x:.2f} {u}\nReasoning: pixel ratio 3.4', '',
        'I cannot determine this from the video.', 'N/A', '0', f'{x:.2f} ± 0.3 {u}', f'is: {x:.2f}',
        f'Example: {x:.1f} cm', f'{x:.2f} kilometers', f'{x:.2f} m/s/s', f'.{int(x * 10) % 10} {u}',
        f'about {x:.0f}', f'{x:.2f} meters per second', f'Time: 3 s, distance {x:.2f} {u}',
    ]


def same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def main() -> None:
    warnings.filterwarnings('ignore')
    root = Path(os.environ['LMUData']) / 'QuantiPhy'
    off_dir = fetch_official(root)
    ns = official_namespace(off_dir / 'model_run_example' / 'run_API_results.py')
    import vlmeval.dataset as D
    from vlmeval.dataset import quantiphy as qp
    print(f'vlmeval imported from {Path(D.__file__).parent.parent}; official code {COMMIT[:7]} in {off_dir}')

    gt_csv = pd.read_csv(off_dir / 'quantiphy_validation.csv')
    ds = D.QuantiPhy('QuantiPhy')
    data = ds.data.copy()
    check(len(data) == 158 and list(data['index']) == list(gt_csv.iloc[:, 0]), 'TSV rows/ids == official GT file')
    check(np.allclose(data['answer'].astype(float), gt_csv['ground_truth_posterior']), 'TSV answers == official GT')

    # 1. prompt parity
    mism = [i for i, (_, r) in enumerate(gt_csv.iterrows())
            if qp.build_user_text(r['question'], r['ground_truth_prior'], r['depth_info'], r['video_type'])
            != official_user_text(ns, r)]
    check(not mism, f'prompt parity: user text == official zero-shot v3 text on {len(gt_csv)} rows '
          f'({len(mism)} mismatches)')
    tsv_text = [qp.build_user_text(r['question'], r['prior'], r['depth_info'], r['video_type'])
                for _, r in data.iterrows()]
    check(tsv_text == [official_user_text(ns, r) for _, r in gt_csv.iterrows()],
          'prompt parity: user text built from the TSV == official text from quantiphy_validation.csv')
    check(qp.SYSTEM_PROMPT in (off_dir / 'model_run_example' / 'run_API_results.py').read_text().replace(
        '"\n            "', '').replace('\\n', '\n'), 'system prompt == official text')

    # 2. parse parity
    rng = np.random.default_rng(0)
    responses = []
    for gt in data['answer']:
        responses += synthetic_responses(gt, rng)
    n_mis = sum(not same(qp.parse_number(r), ns['parse_number'](r)) for r in responses)
    n_nan = sum(math.isnan(ns['parse_number'](r)) for r in responses)
    check(n_mis == 0, f'parse parity: {len(responses)} synthetic responses, {n_mis} mismatches '
          f'({n_nan} unparseable in both)')
    # parse_number_span (lenient unit binding) must select the same value as the official parser, and its span
    # must cover exactly that number in the text (after the exponent normalisation lenient_value applies)
    n_span = 0
    for r in responses:
        t = qp._EXP_RE.sub(r'\1²', r)
        v, span = qp.parse_number_span(t)
        ok = same(v, ns['parse_number'](t))
        if ok and span is not None:
            frag = t.strip()[span[0]:span[1]]
            nums = qp.num_re.findall(frag)
            ok = bool(nums) and abs(float(nums[-1])) == v
        n_span += not ok
    check(n_span == 0, f'parse_number_span: value == official parse_number and span covers that number on '
          f'{len(responses)} responses ({n_span} mismatches)')

    # 3. scorer parity with the official evaluator.py (unmodified, subprocess)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sets = {}
        for k in range(4):
            r = np.random.default_rng(k + 1)
            preds = []
            for gt in data['answer']:
                cand = synthetic_responses(gt, r)
                preds.append(cand[r.integers(len(cand))])
            sets[f'synthetic{k}'] = preds
        sets['gt'] = [f'{g} m' for g in data['answer']]
        for name, preds in sets.items():
            ev = data.copy()
            ev['prediction'] = preds
            eval_file = td / f'model_{name}_QuantiPhy.xlsx'
            ev.to_excel(eval_file, index=False)
            ours = ds.evaluate(str(eval_file))
            in_dir, out_dir = td / f'in_{name}', td / f'out_{name}'
            in_dir.mkdir()
            exported = td / f'model_{name}_QuantiPhy_quantiphy_official.csv'
            os.replace(exported, in_dir / f'{name}.csv')
            subprocess.run([sys.executable, '-I', str(off_dir / 'evaluator.py'), str(in_dir), str(out_dir),
                            '--gt_file', str(off_dir / 'quantiphy_validation.csv')],
                           check=True, capture_output=True, text=True)
            res = pd.read_csv(out_dir / 'all_model_results.csv').iloc[0]
            pairs = [('Overall', 'mra_average')] + [(c, f'mra_{c}') for c in qp.CATEGORIES] + \
                [(k[4:], k) for k in res.index if k.startswith('mra_bg_') or k.startswith('mra_obj_')]
            bad = []
            for ok_, ov in pairs:
                a, b = ours.get(ok_), res[ov]
                if (a is None or (isinstance(a, float) and math.isnan(a))) and pd.isna(b):
                    continue
                if a is None or pd.isna(b) or abs(a - 100 * b) > 1e-9:
                    bad.append((ok_, a, b))
            if abs(ours['invalid_percentage'] - res['invalid_percentage']) > 1e-9:
                bad.append(('invalid_percentage', ours['invalid_percentage'], res['invalid_percentage']))
            check(not bad, f'scorer parity [{name}]: ours Overall {ours["Overall"]:.4f} vs official '
                  f'{100 * res["mra_average"]:.4f}; {len(pairs) + 1} metrics, {len(bad)} mismatches {bad}')
            if name == 'gt':
                check(abs(ours['Overall'] - 100) < 1e-9 and all(abs(ours[c] - 100) < 1e-9 for c in qp.CATEGORIES),
                      'GT as prediction ("<gt> m") -> Overall 100 and 100 per category')

        # 5. garbage
        for name, preds in {'no number': ['I cannot tell.'] * len(data),
                            'random numbers': [f'{v:.2f} m' for v in np.random.default_rng(7).uniform(0, 1000,
                                                                                                     len(data))]}.items():
            ev = data.copy()
            ev['prediction'] = preds
            eval_file = td / f'garbage_{name.replace(" ", "_")}_QuantiPhy.xlsx'
            ev.to_excel(eval_file, index=False)
            ours = ds.evaluate(str(eval_file))
            check(ours['Overall'] < 10, f'garbage ({name}) -> Overall {ours["Overall"]:.2f} (< 10)')

    # 4. published official GPT-5.1 result
    g = pd.read_csv(off_dir / 'model_outputs' / 'gpt-5.1.csv')
    g = g.rename(columns={g.columns[0]: 'id'})
    g['ground_truth_posterior'] = g['id'].map(dict(zip(gt_csv.iloc[:, 0], gt_csv['ground_truth_posterior'])))
    _, summ = qp.score_frame(g, gt_col='ground_truth_posterior')
    pub = pd.read_csv(off_dir / 'mra_results' / 'all_model_results.csv').iloc[0]
    keys = ['mra_average', 'mra_S2', 'mra_D2', 'mra_S3', 'mra_D3', 'invalid_percentage']
    bad = [(k, summ[k], pub[k]) for k in keys if abs(summ[k] - pub[k]) > 1e-9]
    check(not bad, f'official GPT-5.1 outputs: our scorer MRA avg {100 * summ["mra_average"]:.2f} == published '
          f'{100 * pub["mra_average"]:.2f} (S2/D2/S3/D3 {[round(100 * summ[k], 2) for k in keys[1:5]]}); {bad}')

    # 6. parse cases (expected = official behaviour, quirks included)
    cases = [
        ('2.5 cm', 2.5), ('2.5cm', 2.5), ('-3 m/s', 3.0), ('1.2e-2 m', 0.012), ('Answer: 12 cm', 12.0),
        ('Final Answer: 0.7 meters', 0.7), ('The velocity is 2.5 m/s^2', 2.0), ('v = 3.1 m/s', 3.1),
        ('9.81 m/s²', 9.81), ('3 km/h', 3.0), ('Approximately 4.2-4.8 cm', 4.8), ('', float('nan')),
        ('I cannot determine this.', float('nan')), ('0', 0.0), ('.5 m', 0.5),
        ('1,250 mm', 250.0),           # thousands separator: official takes the last number
        ('At t=2s: 5 m', 5.0), ('5 m at t=2s', 2.0),  # delimiter '=' -> text after it ("2s"), last number
        ('2.5 ± 0.3 cm', 0.3),
        # official quirk: unescaped '^' in EXACT_NUMBER_WITH_UNIT_RE + 'm'/'s' stripped before 'm/s^2'
        ('980 cm/s^2', 2.0), ('9.8 m/s2 downward', 2.0), ('9.8 m/s2', 9.8),
    ]
    for text, exp in cases:
        got = qp.parse_number(text)
        check(same(got, exp) and same(got, ns['parse_number'](text)),
              f'parse {text!r} -> {got} (expected {exp}, official {ns["parse_number"](text)})')
    conv = [
        ('0.25 m', 'What is the length of the block in cm?', 25.0),
        ('250 mm', 'What is the length of the block in cm?', 25.0),
        ('25 cm', 'What is the length of the block in cm?', 25.0),
        ('1.5 m/s', 'What is the velocity of the ball at 1s in cm/s?', 150.0),
        ('980 cm/s^2', 'What is the acceleration of the ball in m/s^2?', 9.8),
        ('980 cm/s^2', 'What is the acceleration of the ball in cm/s^2?', 980.0),
        ('The acceleration is 2.5 m/s^2', 'What is the acceleration of the ball in m/s^2?', 2.5),
        ('9.8 m/s2 downward', 'What is the acceleration of the ball in m/s^2?', 9.8),
        ('0.98 m/s²', 'What is the acceleration of the ball in cm/s^2?', 98.0),
        ('36 km/h', 'What is the speed of the car in m/s?', 10.0),
        ('3 ft', 'What is the height of the box in meters?', 0.9144),
        ('2 m/s', 'What is the length of the block in cm?', 2.0),  # dimension mismatch -> unchanged
        ('42', 'What is the length of the block in cm?', 42.0),     # no unit -> unchanged
        # unit bound to the occurrence the parser selected, not to another occurrence of the same number
        ('0.5 m wide. Final Answer: 0.5', 'What is the width of the box in cm?', 0.5),
        ('2 m, so the answer is: 2', 'What is the length of the block in cm?', 2.0),
        ('Answer: 50 cm (0.5 m)', 'What is the width of the box in cm?', 50.0),
        ('0.5 cm? No. Final Answer: 0.5 m', 'What is the width of the box in cm?', 50.0),
        ('  The speed is 3.6 km/h', 'What is the speed of the car in m/s?', 1.0),
    ]
    for text, q, exp in conv:
        got = qp.lenient_value(text, q)
        check(abs(got - exp) < 1e-9, f'lenient {text!r} for "{q[-12:]}" -> {got} (expected {exp})')

    # 7. test split vs the official submission template (ids 1..3289 = parquet row order)
    tsv = Path(os.environ['LMUData']) / 'QuantiPhy_test.tsv'
    if tsv.exists():
        tpl = root / 'competition' / 'quantiphy_submission_template.csv'
        if not tpl.exists():
            tpl.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(qp.TEMPLATE_URL, timeout=60) as r:
                tpl.write_bytes(r.read())
        check(hashlib.sha256(tpl.read_bytes()).hexdigest() == qp.TEMPLATE_SHA256, 'submission template sha256 pinned')
        t = pd.read_csv(tpl)
        d = pd.read_csv(tsv, sep='\t')
        check(list(t.columns) == qp.SUBMISSION_COLUMNS and list(t['id']) == list(d['index'])
              and list(t['video_id']) == list(d['video_id'])
              and list(t['inference_type']) == list(d['inference_type']),
              f'QuantiPhy_test ids / video_id / inference_type == submission template ({len(t)} rows)')
        dt = D.QuantiPhy('QuantiPhy_test')
        with tempfile.TemporaryDirectory() as td:
            ev = dt.data.copy()
            ev['prediction'] = ['3.5 m/s'] * (len(ev) - 1) + ['no idea']
            ef = Path(td) / 'model_QuantiPhy_test.xlsx'
            ev.to_excel(ef, index=False)
            res = dt.evaluate(str(ef))
            sub = pd.read_csv(Path(td) / 'model_QuantiPhy_test_quantiphy_submission.csv')
            check(list(sub.columns) == list(t.columns) and list(sub['id']) == list(t['id'])
                  and sub['parsed_value'].iloc[:-1].eq(3.5).all() and pd.isna(sub['parsed_value'].iloc[-1])
                  and res['n'] == len(t) and 'Overall' not in res,
                  f'QuantiPhy_test evaluate -> submission CSV with template columns/ids, no score ({dict(res)})')
    else:
        print('[SKIP] QuantiPhy_test.tsv not built; template check skipped')

    if FAILS:
        print(f'\nFAIL ({len(FAILS)}):\n  ' + '\n  '.join(FAILS))
        raise SystemExit(1)
    print('\nOK: QuantiPhy prompt, parser and scorer match the official code.')


if __name__ == '__main__':
    main()
