#!/usr/bin/env python
"""VSTI-Bench scorer tests (offline, no GPU; needs $LMUData/VSTIBench/test.json and the official task dir).

  1. GT as prediction -> Overall = Overall_code9 = 100 (also lenient).
  2. Garbage -> 0; constant 'A' / random letters + random numbers -> low (reported vs chance).
  3. Parity with the official scorer: the official VITA-Group/VLM-3R
     thinking-in-space/lmms_eval/tasks/vstibench/utils.py (@ 206afd0) is imported unchanged from
     --official-dir (which must contain utils.py and vstibench.yaml) and run on a synthetic response
     set (several templates per question: letters with/without '.', '(B)', 'The answer is B', lower
     case, numbers with units / commas / trailing '.', nan, inf, empty, ...). Per-question
     vsibench_process_results scores must equal ours (0 mismatches) and vsibench_aggregate_results
     must equal our Overall_code9. The paper Avg (Overall) is checked against an independent
     group computation.
  4. Hand-made numeric / MCQ parse cases (official and lenient).

    python scripts/test_vstibench_scoring.py --official-dir <VLM-3R>/thinking-in-space/lmms_eval/tasks/vstibench
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def load_official(d: Path):
    with tempfile.TemporaryDirectory() as td:
        for f in ("utils.py", "vstibench.yaml"):
            shutil.copy(d / f, Path(td) / f)
        spec = importlib.util.spec_from_file_location("vsti_official_utils", Path(td) / "utils.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    return mod


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official-dir", required=True)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    from vlmeval.dataset import vstibench as V

    off = load_official(Path(args.official_dir))
    docs = json.load(open(Path(os.environ["LMUData"]) / "VSTIBench" / "test.json"))
    df = pd.DataFrame(docs)
    df["options_s"] = [json.dumps(o) if isinstance(o, list) else "" for o in df["options"]]
    fails = []

    def ours(preds):
        t = pd.DataFrame({"question_type": df["question_type"], "prediction": preds,
                          "mc_answer": [a if isinstance(a, str) else "" for a in df["mc_answer"]],
                          "ground_truth": df["ground_truth"].astype(str), "options": df["options_s"]})
        return V.score_dataframe(t)

    def check(name, cond, msg=""):
        print(f"[{'OK' if cond else 'FAIL'}] {name} {msg}")
        if not cond:
            fails.append(name)

    # 1-2. GT / garbage / chance
    gt = [d["mc_answer"] if d["question_type"] in V.MCA_QUESTION_TYPES else d["ground_truth"] for d in docs]
    _, s, _ = ours(gt)
    check("GT -> 100", abs(s["Overall"] - 100) < 1e-9 and abs(s["Overall_code9"] - 100) < 1e-9
          and abs(s["Overall_lenient"] - 100) < 1e-9, f"Overall {s['Overall']:.2f} code9 {s['Overall_code9']:.2f}")
    _, s, _ = ours(["I don't know"] * len(df))
    check("garbage -> 0", s["Overall"] == 0 and s["Overall_code9"] == 0 and s["Overall_lenient"] == 0,
          f"Overall {s['Overall']:.2f}")
    rng = np.random.default_rng(0)
    nopt = [len(o) if isinstance(o, list) else 0 for o in df["options"]]
    rnd = ["ABCD"[rng.integers(n)] if n else f"{rng.uniform(0.2, 8.2):.1f}" for n in nopt]
    _, s, _ = ours(rnd)
    chance_groups = [np.mean([np.mean([1 / n for n, q in zip(nopt, df["question_type"]) if q == qq])
                              for qq in qs]) * 100 for t, qs in V.TASKS.items() if qs[0] in V.MCA_QUESTION_TYPES]
    check("random -> low", s["Overall"] < 40, f"Overall {s['Overall']:.2f} (MCQ groups "
          + ", ".join(f"{t} {s[t]:.1f}" for t in V.TASKS) + f"; expected MCQ chance groups "
          + ", ".join(f"{c:.1f}" for c in chance_groups) + ")")
    _, s, _ = ours(["A"] * len(df))
    check("all 'A' -> low", s["Overall"] < 40, f"Overall {s['Overall']:.2f}")

    # 3. Parity with the official scorer on a synthetic response set
    mcq_t = ["{a}", "{a}.", "({a})", "The answer is {a}.", "{al}", "{w}", "{a} because ...", "", "{a}\n", " {a}",
             "Answer: {a}", "{opt}", "{a}. {txt}", "**{a}**"]
    na_t = ["{v}", "{v}.", "{v} meters", "{v}m", "about {v}", "{v:,.0f}", "nan", "inf", "-{v}", "0", "",
            "{v}\n", " {v}", "{v2}", "1e1", "{v}.0.", "approximately {v} m", "{v} ", "The distance is {v}"]
    preds, kinds = [], []
    for d in docs:
        g = d["ground_truth"]
        if d["question_type"] in V.MCA_QUESTION_TYPES:
            a = d["mc_answer"] if rng.random() < .5 else "ABCD"[rng.integers(len(d["options"]))]
            t = mcq_t[rng.integers(len(mcq_t))]
            p = t.format(a=a, al=a.lower(), w="ABCD"[(("ABCD".index(a)) + 1) % len(d["options"])],
                         opt=d["options"][0], txt=g)
        else:
            v = float(g) * float(rng.choice([1, 1, 0.5, 0.8, 1.2, 1.5, 2, 3, 10]))
            t = na_t[rng.integers(len(na_t))]
            p = t.format(v=round(v, 2), v2=round(v * 1000, 0))
        preds.append(p)
        kinds.append(t)
    scored, s, table = ours(preds)
    off_scores, results = [], []
    for d, p in zip(docs, preds):
        r = off.vsibench_process_results(dict(d), [p])["vsibench_score"]
        key = "accuracy" if d["question_type"] in off.MCA_QUESTION_TYPES else "MRA:.5:.95:.05"
        off_scores.append(float(r[key]))
        results.append(r)
    off_agg = off.vsibench_aggregate_results(results)
    mism = [(i, preds[i], off_scores[i], scored["score"].iloc[i]) for i in range(len(docs))
            if not (off_scores[i] == scored["score"].iloc[i])]
    check("parity per question (official vsibench_process_results)", not mism,
          f"{len(mism)} mismatches / {len(docs)}; templates {len(mcq_t)} MCQ + {len(na_t)} numeric"
          + (f", e.g. {mism[:3]}" if mism else ""))
    check("parity aggregate (official vsibench_aggregate_results == Overall_code9)",
          abs(off_agg - s["Overall_code9"]) < 1e-9, f"official {off_agg:.6f} ours {s['Overall_code9']:.6f}")
    per_q = pd.DataFrame({"q": df["question_type"], "s": off_scores}).groupby("q")["s"].mean()
    paper_avg = np.mean([np.mean([per_q[q] for q in qs]) for qs in V.TASKS.values()]) * 100
    check("paper Avg (mean of 5 groups of sub-type means) == Overall", abs(paper_avg - s["Overall"]) < 1e-9,
          f"independent {paper_avg:.4f} ours {s['Overall']:.4f}; lenient {s['Overall_lenient']:.4f}; "
          f"fallback {int((scored['parse_mode'] == 'fallback').sum())}, unparsed {int((scored['parse_mode'] == 'fail').sum())}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 4. Hand-made parse cases: (qtype, prediction, gt/mc, official, lenient)
    na, mc = "camera_displacement", "camera_movement_direction"
    cases = [
        (na, "2.0", "2.0", 1.0, 1.0), (na, "2", "2.0", 1.0, 1.0), (na, "2.0.", "2.0", 1.0, 1.0),
        (na, "2.0 meters", "2.0", 1.0, 1.0), (na, "2.0m", "2.0", 0.0, 1.0), (na, "about 2", "2.0", 0.0, 1.0),
        (na, "3.0", "2.0", 0.1, 0.1), (na, "2.4", "2.0", 0.7, 0.7), (na, "1.0", "2.0", 0.1, 0.1),
        (na, "nan", "2.0", 0.0, 0.0), (na, "inf", "2.0", 0.0, 0.0), (na, "", "2.0", 0.0, 0.0),
        (na, "-2", "2.0", 0.0, 0.0), (na, "2,0", "2.0", 0.0, 0.0), (na, "1e1", "2.0", 0.0, 0.0),
        (mc, "A", "A", 1.0, 1.0), (mc, "a.", "A", 1.0, 1.0), (mc, "(A)", "A", 0.0, 1.0),
        (mc, "The answer is A", "A", 0.0, 1.0), (mc, "B", "A", 0.0, 0.0), (mc, "A, B", "A", 0.0, 0.0),
    ]
    bad = []
    for qt, p, g, e_off, e_len in cases:
        o = V.official_score(qt, p, g if qt == mc else "", g)
        L, _, mode = V.lenient_score(qt, p, g if qt == mc else "", g, '["A. Forward", "B. Left", "C. Right"]')
        if not (math.isclose(o, e_off, abs_tol=1e-9) and math.isclose(L, e_len, abs_tol=1e-9)):
            bad.append((qt, p, g, o, e_off, L, e_len, mode))
    check("hand-made parse cases", not bad, f"{len(cases) - len(bad)}/{len(cases)}" + (f" bad {bad}" if bad else ""))

    # 5. evaluation_scope / partial headline
    ids, qts = df["id"].tolist(), df["question_type"].tolist()
    sc = V.evaluation_scope(ids, qts, ids)
    check("scope full release", sc["evaluation_scope"] == "full" and not sc["headline_partial"], str(sc))
    sc = V.evaluation_scope(ids[:-1], qts[:-1], ids)
    check("scope one question missing", sc["evaluation_scope"] == "subset" and sc["headline_partial"], str(sc))
    keep = [q != "camera_obj_rel_dist_v1" for q in qts]
    sc = V.evaluation_scope([i for i, k in zip(ids, keep) if k], [q for q, k in zip(qts, keep) if k], ids)
    check("scope one type missing", sc["evaluation_scope"] == "subset" and sc["n_types"] == 8, str(sc))
    sc = V.evaluation_scope(ids + ids[:1], qts + qts[:1], ids)
    check("scope duplicate row", sc["evaluation_scope"] == "subset" and sc["n_duplicate_rows"] == 1, str(sc))

    # 6. zero numeric targets: the official MRA raises ZeroDivisionError (not caught upstream); the
    #    release has none, our port raises the same way.
    nz = V.n_zero_targets(df["question_type"], df["ground_truth"])
    try:
        V.official_score("camera_displacement", "1.0", "", "0")
        raised = False
    except ZeroDivisionError:
        raised = True
    try:
        off.vsibench_process_results(dict(docs[0], question_type="camera_displacement", ground_truth="0"), ["1.0"])
        off_raised = False
    except ZeroDivisionError:
        off_raised = True
    check("zero targets", nz == 0 and raised and off_raised,
          f"{nz} zero targets in the release; target 0 raises ZeroDivisionError: ours {raised}, official {off_raised}")

    # 7. cache manifest: valid as built; tampered pins / video hash / TSV are rejected (non-destructive copies)
    root = Path(os.environ["LMUData"]) / "VSTIBench"
    tsv = Path(os.environ["LMUData"]) / "VSTIBench.tsv"
    man = Path(os.environ["LMUData"]) / "VSTIBench.manifest.json"
    r = V.validate_cache(str(root), str(tsv), str(man))
    check("manifest valid", r is None, str(r))
    m0 = json.loads(man.read_text())
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        def tampered(mod):
            m = json.loads(json.dumps(m0))
            mod(m)
            p = Path(td) / "m.json"
            p.write_text(json.dumps(m))
            return V.validate_cache(str(root), str(tsv), str(p))
        v0 = sorted(m0["videos"])[0]
        cases7 = {
            "revision": lambda m: m.update(hf_revision="27ff0a2ddb5b22c1eb036a4249345e6f00ee8ea1"),
            "build_version": lambda m: m.update(build_version=0),
            "tsv_md5": lambda m: m.update(tsv_md5="0" * 32),
            "video sha256": lambda m: m["videos"][v0].__setitem__(1, "0" * 64),
            "video size": lambda m: m["videos"][v0].__setitem__(0, 1),
            "video list": lambda m: m["videos"].pop(v0),
        }
        res7 = {k: tampered(f) for k, f in cases7.items()}
        tsv2 = Path(td) / "VSTIBench.tsv"
        tsv2.write_text(tsv.read_text() + "\n")
        res7["edited TSV"] = V.validate_cache(str(root), str(tsv2), str(man))
    check("manifest tamper rejected", all(v is not None for v in res7.values()),
          "; ".join(f"{k}: {v}" for k, v in res7.items()))

    if fails:
        print("FAIL:", fails)
        raise SystemExit(1)
    print("\nOK: VSTI-Bench scorer == official on all checks.")


if __name__ == "__main__":
    main()
