#!/usr/bin/env python
"""Offline PhysBench scorer tests (no GPU, no server, no media; needs network once for the labels).

1. GT-as-prediction -> 100 (official headline, every domain / sub-task / ability).
2. Garbage predictions -> low (official rule: only the 17 empty-label rows score, 0.17).
3. Parity: on a synthetic response set (20 templates per item over all 10,002 items) our per-row
   decisions and the aggregated tables equal
     a) the verbatim official scorer (eval/print_test_score.py calculate_accuracy, EvalAI-equivalent),
     b) lmms-eval tasks/physbench @ 9134731 (process_results + aggregate, reproduced below verbatim
        apart from logging).
   Also runs PhysBench.evaluate() end to end on an xlsx prediction file (asserts inside evaluate()
   compare with the official scorer again).
4. PhysBench_val: 200 rows; val_answer.json answers equal test_answer.json (14 rows differ in taxonomy).

    PYTHONPATH=third_party/vlmevalkit-cosmos python scripts/test_physbench_scoring.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import types
import warnings
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')


# ---- lmms-eval lmms_eval/tasks/physbench/utils.py @ 9134731f (verbatim logic, logging removed) ----
def lmms_process_docs(rows):
    return [r for r in rows if bool(r.get('answer', '').strip())]


def lmms_process_results(doc, results):
    pred = results[0].strip() if results else ""
    pred_ans = pred[0].upper() if pred and pred[0].upper() in "ABCD" else ""
    gt_ans = doc.get("answer", "")
    score = 1.0 if pred_ans.upper() == gt_ans.upper() and gt_ans else 0.0
    return {"physbench_accuracy": {"idx": doc.get("idx"), "task_type": doc.get("task_type", ""),
                                   "pred_answer": pred_ans, "answer": gt_ans, "score": score}}


def lmms_aggregate(results):
    task = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total = 0
    for r in results:
        total += 1
        total_correct += r["score"]
        if r.get("task_type", ""):
            task[r["task_type"]]["total"] += 1
            task[r["task_type"]]["correct"] += r["score"]
    overall = 100 * total_correct / total if total > 0 else 0
    return overall, {k: 100 * v["correct"] / v["total"] for k, v in task.items()}
# ----------------------------------------------------------------------------------------------------


TEMPLATES = [
    '{L}', '{L}.', ' {L}', '{l}', '({L})', '{L}) option', 'The answer is {L}.', 'Answer: {L}',
    '{L}\nBecause the ball falls.', '**{L}**', '', 'E', 'I think {L}', '{L}, {M}', 'option {L}',
    '<answer>{L}</answer>', '{M}', 'None of the above', '1', '{L}{M}',
]


def main() -> None:
    os.environ.setdefault('LMUData', '/data0/woosung/datasets/Cosmos3-Reasoner/LMUData')
    import pandas as pd
    import vlmeval
    from vlmeval.dataset import physbench as pb
    from vlmeval.dataset.utils.physbench_official import calculate_accuracy
    from vlmeval.smp import dump
    print('vlmeval from', vlmeval.__file__)

    root = pb.physbench_root()
    label_path = pb._ensure_file(root, 'test_answer.json')
    val_path = pb._ensure_file(root, 'val_answer.json')
    labels = json.load(open(label_path, encoding='utf-8'))
    val = json.load(open(val_path, encoding='utf-8'))
    by_idx = {x['idx']: x for x in labels}
    assert len(labels) == 10002 and len(val) == 200
    assert all(by_idx[v['idx']]['answer'] == v['answer'] for v in val), 'val answers != test_answer.json'
    n_tax = sum(any(by_idx[v['idx']][k] != v[k] for k in ('task_type', 'sub_type', 'ability_type')) for v in val)
    print(f'val_answer.json: 200 ids, answers == test_answer.json; {n_tax} rows with a different taxonomy')
    tmp = Path(tempfile.mkdtemp(prefix='pbtest_', dir=os.environ.get('TMPDIR')))
    fails = []

    def ours(preds, rows):
        gts = [r['answer'] for r in rows]
        hit, len_hit, lmms, lmms_hit = pb.score_rows(preds, gts)
        df = pd.DataFrame(dict(hit=hit, hit_lenient=len_hit, lmms_hit=lmms_hit,
                               has_label=[bool(g) for g in gts],
                               task_type=[r['task_type'] for r in rows], sub_type=[r['sub_type'] for r in rows],
                               ability_type=[r['ability_type'] for r in rows]))
        return pb.summarize(df), hit, lmms

    def official(preds, rows, lab_path, tag):
        sub = tmp / f'{tag}.json'
        json.dump([{'idx': r['idx'], 'answer': p} for r, p in zip(rows, preds)], open(sub, 'w'))
        return calculate_accuracy(str(lab_path), str(sub))

    def compare(tag, preds, rows, lab_path):
        s, hit, lmms = ours(preds, rows)
        off = official(preds, rows, lab_path, tag)
        mism = 0
        for r, p, h in zip(rows, preds, hit):
            if pb.official_hit(p, r['answer']) != h or \
                    (p == r['answer'] or p.startswith(r['answer'])) != h:
                mism += 1
        if abs(off['overall_accuracy'] - s['Overall']) > 1e-9:
            mism += 1
        for kind, pre in (('task_type_accuracy', ''), ('sub_type_accuracy', None), ('ability_type_accuracy', 'ability/')):
            for k, v in off[kind].items():
                if pre is None:
                    key = next(f'{t}/{k}' for t, subs in pb.SUB_ORDER.items() if k in subs)
                else:
                    key = pre + k
                if abs(v['accuracy'] - s[key]) > 1e-9:
                    mism += 1
                    print('  table mismatch', tag, key, v, s[key])
        docs = lmms_process_docs(rows)
        doc_pred = {r['idx']: p for r, p in zip(rows, preds)}
        res = [lmms_process_results(d, [doc_pred[d['idx']]])['physbench_accuracy'] for d in docs]
        l_over, l_task = lmms_aggregate(res)
        our_l = dict(zip([r['idx'] for r in rows], lmms))
        lm = sum(r['pred_answer'] != our_l[r['idx']] for r in res)
        if abs(l_over - s['lmms_overall']) > 1e-9 or s['lmms_n'] != len(docs):
            lm += 1
        for t, v in l_task.items():
            if abs(v - s[f'lmms_{t}']) > 1e-9:
                lm += 1
        print(f'[{tag}] official {off["overall_accuracy"]:.4f} ours {s["Overall"]:.4f} '
              f'({off["overall_correct"]}/{off["overall_total"]}) official mismatches={mism}; '
              f'lmms {l_over:.4f} ours {s["lmms_overall"]:.4f} (n={len(docs)}) lmms mismatches={lm}; '
              f'lenient {s["Overall_lenient"]:.2f}')
        if mism or lm:
            fails.append(f'{tag}: {mism} official / {lm} lmms mismatches')
        return s

    # 1. GT as prediction
    s = compare('gt', [r['answer'] for r in labels], labels, label_path)
    bad = {k: v for k, v in s.items() if isinstance(v, float) and not k.startswith('lmms') and v != 100.0}
    if bad:
        fails.append(f'gt != 100: {bad}')
    s = compare('gt_val', [r['answer'] for r in val], val, val_path)
    if s['Overall'] != 100.0:
        fails.append('val gt != 100')

    # 2. garbage
    s = compare('garbage', ['I cannot tell.'] * len(labels), labels, label_path)
    if s['Overall'] > 1.0:
        fails.append(f'garbage too high: {s["Overall"]}')
    s = compare('all_A', ['A'] * len(labels), labels, label_path)

    # 3. synthetic response set
    rng = random.Random(0)
    preds = []
    for r in labels:
        g = r['answer'].strip()[:1].upper() if r['answer'].strip()[:1].upper() in 'ABCD' else 'A'
        L = g if rng.random() < 0.5 else rng.choice('ABCD')
        M = rng.choice('ABCD')
        preds.append(rng.choice(TEMPLATES).format(L=L, l=L.lower(), M=M))
    s = compare('synthetic', preds, labels, label_path)

    # end-to-end evaluate() on an xlsx prediction file (as run.py writes it)
    df = pd.DataFrame(dict(index=[r['idx'] for r in labels], prediction=preds,
                           question=['q'] * len(labels)))
    ef = tmp / 'model_PhysBench.xlsx'
    dump(df, str(ef))
    fake = types.SimpleNamespace(label_file='test_answer.json', expected=10002, dataset_name='PhysBench')
    out = pb.PhysBench.evaluate(fake, str(ef))
    if abs(out['Overall'] - s['Overall']) > 1e-9 or out['evaluation_scope'] != 'full':
        fails.append(f'evaluate() {out["Overall"]} != {s["Overall"]} / scope {out["evaluation_scope"]}')
    print(f'evaluate(): Overall {out["Overall"]:.4f} official {out["official_overall"]:.4f} scope {out["evaluation_scope"]}')

    # counts
    from collections import Counter
    print('domains', dict(Counter(r['task_type'] for r in labels)))
    print('empty labels', sum(not r['answer'] for r in labels),
          'non-ABCD labels', [(r['idx'], r['answer']) for r in labels if r['answer'] and r['answer'] not in 'ABCD'])

    if fails:
        print('\nFAIL:\n  ' + '\n  '.join(fails))
        sys.exit(1)
    print('\nOK: GT -> 100, garbage -> low, 0 mismatches vs official and lmms-eval scorers.')


if __name__ == '__main__':
    main()
