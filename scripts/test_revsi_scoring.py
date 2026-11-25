#!/usr/bin/env python
"""Offline tests of the ReVSI scorer (vlmeval/dataset/revsi.py). No server, no videos.

Needs only the ReVSI parquet subsets under $LMUData/ReVSI (downloaded by the dataset class).

  1. ground truth as prediction        -> 100 on every task and on Avg (headline=official and lenient)
  2. ground truth wrapped in text ("The answer is ...") -> headline (official) 0, lenient 100
  3. garbage text                      -> 0
  4. random option letters / random numbers -> chance level (MCQ ~ 1/#options, NQ low)
  5. hand-made parse cases (official/headline vs lenient)
  6. MRA unit checks and agreement with upstream's formula
  7. Avg = mean over the present task columns (16-frame subset has 5 tasks)

    python scripts/test_revsi_scoring.py            # or: pytest scripts/test_revsi_scoring.py
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "vlmevalkit-cosmos"))
warnings.filterwarnings("ignore")

from vlmeval.dataset import revsi as R  # noqa: E402

SUBSETS = {"ReVSI_16frame": "16_frame", "ReVSI_32frame": "32_frame", "ReVSI_64frame": "64_frame"}


def load_subset(subset: str) -> pd.DataFrame:
    root = Path(os.environ.get("LMUData", Path.home() / "LMUData")) / "ReVSI"
    p = root / subset / "test-00000-of-00001.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}; run the ReVSI dataset prepare step (or snapshot_download) first")
    df = pd.read_parquet(p)
    return pd.DataFrame({
        "index": df["id"], "question_type": df["question_type"], "answer": df["ground_truth"].astype(str),
        "options": df["options"].apply(R._serialize_options),
    })


def test_ground_truth_is_100():
    for key, subset in SUBSETS.items():
        df = load_subset(subset)
        df["prediction"] = df["answer"]
        _, summ, table = R.score_dataframe(df)
        assert math.isclose(summ["Overall"], 100.0) and math.isclose(summ["Overall_lenient"], 100.0), (key, summ)
        assert all(math.isclose(v, 100.0) for v in table["score"]), (key, table)
        assert all(math.isclose(v, 100.0) for v in table["score_lenient"]), (key, table)
        assert table["n_fallback"].sum() == 0 and table["n_unparsed"].sum() == 0
        n_tasks = len(table) - 1
        assert n_tasks == (5 if subset == "16_frame" else 7), (key, n_tasks)
        print(f"[OK] {key}: GT -> Overall {summ['Overall']:.2f} / lenient {summ['Overall_lenient']:.2f}, "
              f"{len(df)} questions, {n_tasks} tasks")


def test_wrapped_ground_truth():
    df = load_subset("64_frame")
    df["prediction"] = ["The answer is " + a + "." if i % 2 else "<think>hmm 7 things</think>\nAnswer: " + a
                        for i, a in enumerate(df["answer"])]
    _, summ, _ = R.score_dataframe(df)
    assert summ["Overall"] == 0.0, summ
    assert math.isclose(summ["Overall_lenient"], 100.0), summ
    print(f"[OK] wrapped GT text -> headline (official) {summ['Overall']:.2f}, lenient {summ['Overall_lenient']:.2f}")


def test_garbage_is_zero():
    df = load_subset("64_frame")
    for garbage in ["not an answer", "", "nan", "I cannot determine this from the video"]:
        df["prediction"] = garbage
        _, summ, table = R.score_dataframe(df)
        assert summ["Overall"] == 0.0 and summ["Overall_lenient"] == 0.0, (garbage, summ)
    df["prediction"] = np.nan  # empty prediction (e.g. truncated thinking reply)
    _, summ, _ = R.score_dataframe(df)
    assert summ["Overall"] == 0.0 and summ["Overall_lenient"] == 0.0
    print("[OK] garbage / empty predictions -> 0")


def test_random_is_low():
    rng = np.random.default_rng(0)
    df = load_subset("64_frame")
    preds = []
    for _, r in df.iterrows():
        if r["question_type"] in R.MCQ_QUESTION_TYPES:
            preds.append(rng.choice(list(R._option_letters(r["options"]))))
        else:
            preds.append(f"{rng.uniform(0, 20):.2f}")
    df["prediction"] = preds
    _, summ, table = R.score_dataframe(df)
    tab = table.set_index("key")
    for task in ["object_rel_distance", "object_rel_direction", "route_planning"]:
        n_opt = np.mean([len(R.parse_options(o)) for o, q in zip(df["options"], df["question_type"])
                         if q in R.TASKS[task]])
        assert abs(tab.loc[task, "score"] - 100 / n_opt) < 6, (task, tab.loc[task, "score"], n_opt)
    assert summ["Overall"] < 30, summ
    assert math.isclose(summ["Overall"], summ["Overall_lenient"])
    print("[OK] random letters / U(0,20) numbers -> " +
          ", ".join(f"{r.task} {r.score:.1f}" for r in table.itertuples()))


CASES = [
    # (question_type, prediction, gt, options, expect_official, expect_lenient)
    ("object_counting_single", "4", "4", None, 1.0, 1.0),
    ("object_counting_single", "4.", "4", None, 1.0, 1.0),
    ("object_counting_single", "four", "4", None, 0.0, 1.0),
    ("object_counting_single", "There are 4 chairs.", "4", None, 0.0, 1.0),
    ("object_abs_distance", "2.5 meters", "2.5", None, 1.0, 1.0),
    ("object_abs_distance", "2.5m", "2.5", None, 0.0, 1.0),
    ("object_abs_distance", "Approximately 2.5 m.", "2.5", None, 0.0, 1.0),
    ("object_abs_distance", "<think>first 10, then 3</think>\n2.5", "2.5", None, 0.0, 1.0),
    ("object_size_estimation", "1,200", "1200", None, 0.0, 1.0),
    ("object_size_estimation", "110", "100", None, 0.9 - 0.0, 0.9),  # rel err .10 -> passes .50...90 = 9/10
    ("room_size_estimation_single", "0", "20", None, 0.0, 0.0),
    ("room_size_estimation_single", "inf", "20", None, 0.0, 0.0),
    ("object_rel_distance_closest", "B", "B", ["A. x", "B. y", "C. z", "D. w"], 1.0, 1.0),
    ("object_rel_distance_closest", "b.", "B", ["A. x", "B. y", "C. z", "D. w"], 1.0, 1.0),
    ("object_rel_distance_closest", "B. chair", "B", ["A. x", "B. y", "C. z", "D. w"], 1.0, 1.0),
    ("object_rel_distance_closest", "(B)", "B", ["A. x", "B. y", "C. z", "D. w"], 0.0, 1.0),
    ("object_rel_distance_closest", "The answer is B.", "B", ["A. x", "B. y", "C. z", "D. w"], 0.0, 1.0),
    ("object_rel_direction_forward_easy", "<think>A or C?</think>\nC", "C", ["A. l", "B. r", "C. b"], 0.0, 1.0),
    ("object_rel_direction_forward_easy", "D", "C", ["A. l", "B. r", "C. b"], 0.0, 0.0),
    ("route_planning", "A", "B", ["A. x", "B. y", "C. z", "D. w"], 0.0, 0.0),
]


def test_parse_cases():
    for qt, pred, gt, opts, exp_off, exp_rob in CASES:
        off = R.official_score(qt, pred, gt)
        rob, parsed, mode = R.lenient_score(qt, pred, gt, opts)
        assert math.isclose(off, exp_off, abs_tol=1e-9), (qt, pred, gt, off, exp_off)
        assert math.isclose(rob, exp_rob, abs_tol=1e-9), (qt, pred, gt, rob, exp_rob, parsed, mode)
        if exp_off == 1.0:
            assert mode == "official"
    print(f"[OK] {len(CASES)} hand-made parse cases")


def test_mra():
    up = lambda p, t: float(((abs(p - t) / t) <= (1 - np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05 + 2)))).mean())
    assert len(np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05 + 2))) == 10
    assert R.mean_relative_accuracy(10.0, 10.0) == 1.0
    assert R.mean_relative_accuracy(20.0, 10.0) == 0.0
    assert math.isclose(R.mean_relative_accuracy(15.0, 10.0), 0.1)   # rel err .5 passes only theta=.50
    rng = np.random.default_rng(1)
    for _ in range(2000):
        t = rng.uniform(0.1, 50); p = rng.uniform(0, 100)
        assert R.mean_relative_accuracy(p, t) == up(p, t)
    # Upstream float quirk kept for parity (PROTOCOL §6.1 "upstream quirks"): relative errors that sit
    # exactly on a threshold pass or fail depending on float rounding.
    assert R.mean_relative_accuracy(120.0, 100.0) == 0.6   # exact math: 0.7 (0.2 <= 1-0.8)
    assert R.mean_relative_accuracy(80.0, 100.0) == 0.6    # exact math: 0.7
    assert R.mean_relative_accuracy(1.2, 1.0) == 0.7       # same relative error, other side of the boundary
    print("[OK] MRA thresholds 0.50..0.95 (10), matches upstream formula incl. float-boundary quirk")


def test_summary_keys():
    df = load_subset("64_frame")
    df["prediction"] = df["answer"]
    _, summ, table = R.score_dataframe(df)
    tasks = list(R.TASKS)
    assert list(summ) == ["Overall", "Overall_lenient"] + tasks + [f"{t}_lenient" for t in tasks], list(summ)
    assert list(table.columns[:6]) == ["task", "key", "metric", "n", "score", "score_lenient"]
    print("[OK] summary keys: Overall/<task> (official parser, headline), Overall_lenient/<task>_lenient")


def test_avg_is_task_mean():
    df = load_subset("16_frame")
    rng = np.random.default_rng(2)
    df["prediction"] = [a if rng.random() < 0.5 else "x" for a in df["answer"]]
    _, summ, table = R.score_dataframe(df)
    tasks = table[table["key"] != "overall"]
    assert set(tasks["key"]) == {"object_counting", "object_abs_distance", "object_size_estimation",
                                 "object_rel_distance", "object_rel_direction"}
    assert math.isclose(summ["Overall"], tasks["score"].mean())
    # merged task = mean of sub-type means (not pooled)
    sc, _, _ = R.score_dataframe(df)
    sub = sc[sc["task"] == "object_rel_direction"].groupby("question_type")["score"].mean()
    assert math.isclose(summ["object_rel_direction"], 100 * sub.mean())
    print("[OK] Avg = mean of present task columns; merged tasks = mean of sub-type means")


def main() -> None:
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL OK")


if __name__ == "__main__":
    main()
