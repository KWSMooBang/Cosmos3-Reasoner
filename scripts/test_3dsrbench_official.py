#!/usr/bin/env python
"""3DSRBench: official metric vs. our keys (offline, no GPU / server; needs $LMUData/3DSRBench.tsv).

  1. TSV facts: md5, rows, base ids, flip copies, option orderings, questions after collapsing orderings.
  2. For synthetic prediction sets (GT, garbage, all-A, random seed 0/1, GT with some rotated copies wrong,
     GT with flip copies wrong) writes a run.py-style prediction xlsx and runs
       * ThreeDSRBenchOfficial.evaluate (key 3DSRBench_official),
       * the official authors' scorer compute_3drbench_results_circular.py (HF ccvl/3DSRBench @ 5c90e422)
         VERBATIM via --official-script, on the `_result.xlsx` laid out like its `_openai_result.xlsx`,
       * upstream _3DSRBench.evaluate (key 3DSRBench) -> DataFrame -> run.py's .df.eval.json ->
         cosmos_eval parse_score (= what run_suite reports for that key).
     and checks Overall + 4 types + 12 sub-types of ours == official (0 mismatches at 1e-9 / the
     script's 0.1 rounding), and reports which upstream setting parse_score picks.

    python scripts/test_3dsrbench_official.py --official-script <path>/compute_3drbench_results_circular.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
KIT = REPO / "third_party" / "vlmevalkit-cosmos"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official-script", required=True)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(KIT / "cosmos_eval"))
    import parse_score
    import vlmeval.dataset as D
    from vlmeval.dataset.threedsrbench_official import N_QUESTIONS, N_ROWS, TSV_MD5, TYPES, SUBTYPES
    from vlmeval.smp import md5

    tsv = Path(os.environ["LMUData"]) / "3DSRBench.tsv"
    assert md5(str(tsv)) == TSV_MD5, "TSV md5"
    ours = D.ThreeDSRBenchOfficial("3DSRBench_official")
    up = D._3DSRBench("3DSRBench")
    meta = ours.data
    q = meta["qid"].astype(str)
    print(f"TSV {tsv.name} md5 {TSV_MD5}: {len(meta)} rows; base ids {q.str[:8].nunique()}; "
          f"'-flip' rows {q.str.contains('-flip').sum()}; ordering suffix -1/-2/-3 rows "
          f"{(q.str[-2] == '-').sum()}; questions after collapsing orderings "
          f"{q.map(lambda x: x[:-2] if x[-2] == '-' else x).nunique()}; options/question "
          f"{meta[['A', 'B', 'C', 'D']].notna().sum(1).value_counts().to_dict()}")
    assert len(meta) == N_ROWS
    letters = meta[["A", "B", "C", "D"]].notna().sum(1)

    rng = np.random.default_rng(0)
    gt = meta["answer"].astype(str).tolist()
    is_rot = (q.str[-2] == "-").to_numpy()
    is_flip = q.str.contains("-flip").to_numpy()

    def wrong(a, n):
        return "ABCD"[("ABCD".index(a) + 1) % n]
    sets = {
        "gt": gt,
        "garbage": ["I cannot tell."] * len(meta),
        "all_A": ["A"] * len(meta),
        "random_seed0": ["ABCD"[rng.integers(n)] for n in letters],
        "random_seed1": [f"{c}" for c in (lambda r1: ["ABCD"[r1.integers(n)] for n in letters])(np.random.default_rng(1))],
        "gt_but_30pct_rotations_wrong": [wrong(a, n) if r and rng.random() < .3 else a
                                        for a, n, r in zip(gt, letters, is_rot)],
        "gt_but_30pct_flips_wrong": [wrong(a, n) if f and rng.random() < .3 else a
                                     for a, n, f in zip(gt, letters, is_flip)],
        "text_answers": [f"The answer is ({a})." if rng.random() < .5 else wrong(a, n) for a, n in zip(gt, letters)],
    }
    keys = ["overall"] + TYPES + SUBTYPES
    fails = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        td = Path(td)
        for name, preds in sets.items():
            d = td / name
            d.mkdir()
            pred = meta.drop(columns=["image"]).copy()
            pred["prediction"] = preds
            res = {}
            for key, ds in (("3DSRBench_official", ours), ("3DSRBench", up)):
                ef = d / f"M_{key}.xlsx"
                pred.to_excel(ef, index=False)
                res[key] = ds.evaluate(str(ef), model="exact_matching", nproc=1)
            o = res["3DSRBench_official"]
            # official scorer, verbatim, on its expected file layout
            od = d / "official"
            (od / "outputs" / "M").mkdir(parents=True)
            r = pd.read_excel(d / "M_3DSRBench_official_result.xlsx")
            cols = ["index", "qid", "question", "A", "B", "C", "D", "answer", "category", "image_source",
                    "image_url", "prediction", "hit"]
            r[cols].to_excel(od / "outputs" / "M" / "M_3DSRBenchv1_openai_result.xlsx", index=False)
            shutil.copy(args.official_script, od / "score.py")
            subprocess.run([sys.executable, "-I", "score.py"], cwd=od, check=True, capture_output=True)
            off = pd.read_csv(od / "results_3DSRBenchv1.csv").set_index("model").loc["M"]
            mism = []
            dj = d / "M_3DSRBench_official.dict.eval.json"
            dj.write_text(json.dumps(o))
            rep_o = parse_score.report(eval_json=str(dj), dataset_name="3DSRBench_official")
            if abs(rep_o["overall"] - o["Overall"]) > 0.005 + 1e-9:   # parse_score rounds to 0.01
                mism.append(("parse_score(3DSRBench_official)", rep_o["overall"], o["Overall"]))
            for k in keys:
                ok_ = o["Overall"] if k == "overall" else o[k]
                if abs(round(ok_, 1) - off[k]) > 1e-6:
                    mism.append((k, ok_, off[k]))
            # upstream key through run.py's df.eval.json + parse_score
            df = res["3DSRBench"]
            if len(df) < len(df.columns):
                df = df.T
            ej = d / "M_3DSRBench.df.eval.json"
            ej.write_text(df.to_json(orient="split"))
            rep = parse_score.report(eval_json=str(ej), dataset_name="3DSRBench")
            up_rows = res["3DSRBench"].set_index("setting")["Overall"] * 100
            picked = [s for s, v in up_rows.items() if abs(v - rep["overall"]) < 1e-6]
            print(f"[{'OK' if not mism else 'FAIL'}] {name:30s} official script {off['overall']:5.1f} | ours "
                  f"{o['Overall']:6.2f} (h {o['height']:.1f} l {o['location']:.1f} o {o['orientation']:.1f} "
                  f"m {o['multi_object']:.1f}, n={o['n_questions']}, {o['evaluation_scope']}) | upstream "
                  f"vanilla {up_rows['vanilla']:.2f} flip {up_rows['flip_eval']:.2f} circ {up_rows['circ_eval']:.2f} "
                  f"flip_circ {up_rows['flip_circ_eval']:.2f} | parse_score('3DSRBench') -> {rep['overall']:.2f} "
                  f"(= {picked or 'none'})")
            if o["evaluation_scope"] != "full" or o["headline_partial"] or o["n_incomplete_excluded"]:
                mism.append(("scope", o["evaluation_scope"], o["headline_partial"]))
            if abs(o["upstream_circ_eval"] - up_rows["circ_eval"]) > 1e-9:
                mism.append(("upstream_circ_eval vs upstream key", o["upstream_circ_eval"], up_rows["circ_eval"]))
            if abs(o["upstream_circ_eval"] - o["Overall"]) > 1e-9:
                mism.append(("upstream_circ_eval", o["upstream_circ_eval"], o["Overall"]))
            if mism:
                fails.append((name, mism))
        # --- partial inputs: missing orderings / duplicated copies / missing questions ---
        from vlmeval.dataset.threedsrbench_official import official_group_key, score_validated
        mq, cats = q.tolist(), meta["category"].tolist()
        gkey = [official_group_key(x) for x in mq]
        target = gkey[0]

        def check(name, cond, msg):
            print(f"[{'OK' if cond else 'FAIL'}] {name:30s} {msg}")
            if not cond:
                fails.append((name, msg))
        # (a) one correct copy, its other ordering(s) missing, everything else wrong
        rows = [(x, c, 1 if g == target else 0) for x, c, g in zip(mq, cats, gkey)
                if not (g == target and x != target)]
        r = score_validated(*zip(*rows), mq)
        check("missing ordering", r["overall"] == 0 and r["n_incomplete_excluded"] == 1 and r["headline_partial"]
              and r["evaluation_scope"] == "subset" and r["n_questions"] == N_QUESTIONS - 1
              and r["overall_present_copies"] > 0,
              f"Overall {r['overall']:.4f} (official script on present copies {r['overall_present_copies']:.4f}), "
              f"excluded {r['n_incomplete_excluded']}, scope {r['evaluation_scope']}, n={r['n_questions']}")
        # (b) duplicated copy of a question answered correctly
        rows = [(x, c, 1) for x, c in zip(mq, cats)] + [(mq[0], cats[0], 1)]
        r = score_validated(*zip(*rows), mq)
        check("duplicated copy", r["n_duplicated_excluded"] == 1 and r["n_duplicate_rows"] == 1
              and r["evaluation_scope"] == "subset" and r["n_questions"] == N_QUESTIONS - 1 and r["overall"] == 100,
              f"dup rows {r['n_duplicate_rows']}, excluded {r['n_duplicated_excluded']}, scope {r['evaluation_scope']}")
        # (c) whole questions missing (complete copies of the rest) -> subset, no exclusions
        keepq = set(sorted(set(gkey))[:100])
        rows = [(x, c, 1) for x, c, g in zip(mq, cats, gkey) if g in keepq]
        r = score_validated(*zip(*rows), mq)
        check("missing questions", r["n_questions"] == 100 and r["n_missing_questions"] == N_QUESTIONS - 100
              and r["n_incomplete_excluded"] == 0 and r["evaluation_scope"] == "subset" and r["headline_partial"],
              f"n={r['n_questions']}, missing {r['n_missing_questions']}, scope {r['evaluation_scope']}")
        # (d) unknown qid
        try:
            score_validated(["ZZZZZZZZ"], [cats[0]], [1], mq)
            check("unknown qid", False, "no error")
        except ValueError:
            check("unknown qid", True, "ValueError")
        # (e) end to end: evaluate() on a GT eval file with the '-1' copy of 50 questions dropped
        d = td / "e2e_missing"
        d.mkdir()
        drop = set(sorted({g for g, x in zip(gkey, mq) if x != g})[:50])
        pred = meta.drop(columns=["image"]).copy()
        pred["prediction"] = gt
        pred = pred[[not (g in drop and x.endswith("-1")) for g, x in zip(gkey, mq)]]
        ef = d / "M_3DSRBench_official.xlsx"
        pred.to_excel(ef, index=False)
        o = ours.evaluate(str(ef), model="exact_matching", nproc=1)
        check("e2e missing orderings", o["evaluation_scope"] == "subset" and o["headline_partial"]
              and o["n_incomplete_excluded"] == 50 and o["n_questions"] == N_QUESTIONS - 50 and o["Overall"] == 100,
              f"Overall {o['Overall']:.2f}, n={o['n_questions']}, excluded {o['n_incomplete_excluded']}, "
              f"scope {o['evaluation_scope']}")
    if fails:
        print("FAIL", json.dumps(fails, default=str, indent=1))
        raise SystemExit(1)
    print(f"\nOK: 3DSRBench_official == official compute_3drbench_results_circular.py on {len(sets)} prediction "
          f"sets (overall, 4 types, 12 sub-types); == upstream circ_eval. Questions scored: {N_QUESTIONS}.")


if __name__ == "__main__":
    main()
