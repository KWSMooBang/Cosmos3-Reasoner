#!/usr/bin/env python
"""Collect every per-bench score.json written by run_suite.py into results/summary.csv.

    python scripts/collect.py                       # scans results/raw/**/score.json
    python scripts/collect.py --root /data0/... --out results/summary.csv

One row per (model, suite, bench, mode) with mode = full | limit<N>_seed<S>; if the same
combination was run more than once, the newest run wins. Full and subset rows never overwrite
each other.
`overall` is parse_score.py's headline Overall (0-100). `overall_lenient` is filled only for benches
that report a diagnostic lenient-parser score (ReVSI `Overall_lenient`; PROTOCOL §6.1), else empty.
Sub-scores go into a JSON column.
Rows with a sample_limit are smoke runs and must not be reported as benchmark scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIELDS = ["model", "suite", "domain", "bench", "mode", "sample_limit", "sample_seed", "status",
          "overall", "overall_lenient", "finished", "eval_json", "subscores"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "results" / "raw"))
    ap.add_argument("--out", default=str(REPO / "results" / "summary.csv"))
    args = ap.parse_args(argv)

    latest: dict[tuple, dict] = {}
    for path in sorted(Path(args.root).glob("**/score.json")):
        row = json.loads(path.read_text("utf-8"))
        key = (row.get("model"), row.get("suite"), row.get("bench"), row.get("mode", "full"))
        if key not in latest or row.get("finished", "") >= latest[key].get("finished", ""):
            latest[key] = row

    rows = []
    for key in sorted(latest, key=lambda k: tuple(str(x) for x in k)):
        r = latest[key]
        score = r.get("score")
        lenient = r.get("score_lenient", (r.get("subscores") or {}).get("Overall_lenient"))
        rows.append({
            "model": r.get("model"), "suite": r.get("suite"), "domain": r.get("domain", ""),
            "bench": r.get("bench"), "mode": r.get("mode", "full"),
            "sample_seed": r.get("sample_seed", ""), "status": r.get("status"),
            "overall": f"{score:.2f}" if isinstance(score, (int, float)) else "",
            "overall_lenient": f"{lenient:.2f}" if isinstance(lenient, (int, float)) else "",
            "sample_limit": r.get("sample_limit", ""), "finished": r.get("finished", ""),
            "eval_json": str(Path(r["eval_json"]).relative_to(REPO)) if r.get("eval_json", "").startswith(str(REPO))
            else r.get("eval_json", ""),
            "subscores": json.dumps(r.get("subscores", {}), sort_keys=True),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} row(s) to {out}")


if __name__ == "__main__":
    main()
