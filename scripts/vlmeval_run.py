#!/usr/bin/env python
"""Thin wrapper around the vendored VLMEvalKit `run.py` that adds a per-dataset sample limit.

Stock `run.py` has no sample-limit option. This wrapper imports it unchanged, wraps
`run.build_dataset_from_config` so that, right after the dataset object is built (TSV loaded,
image references resolved), `dataset.data` is replaced by a deterministic random subset, and then
calls `run.main()` with the same CLI arguments.

    VLMEVAL_SAMPLE_LIMIT=20 VLMEVAL_SAMPLE_SEED=0 \
        python scripts/vlmeval_run.py --config cfg.json --work-dir out --verbose --save-eval-results

Must be launched with cwd = third_party/vlmevalkit-cosmos (run_suite.py does that).
Only used for smoke runs: scores on a subset are not comparable to full-benchmark scores, and
`run_suite.py` uses the stock `run.py` whenever no limit is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VLMEVALKIT_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "vlmevalkit-cosmos"


def truncate_dataset(dataset, limit: int, seed: int = 0) -> None:
    """Keep `limit` rows of `dataset.data` (seeded random sample, original order preserved)."""
    data = getattr(dataset, "data", None)
    if data is None or limit <= 0 or len(data) <= limit:
        return
    sub = data.sample(n=limit, random_state=seed).sort_index()
    dataset.data = sub
    # Video datasets iterate over `self.videos` when pack=True; keep it consistent.
    if hasattr(dataset, "videos") and "video" in sub:
        keep = set(sub["video"])
        dataset.videos = [v for v in dataset.videos if v in keep]
    print(f"[vlmeval_run] sample limit: kept {len(sub)}/{len(data)} rows of "
          f"{getattr(dataset, 'dataset_name', '?')} (seed={seed})", flush=True)


def main() -> None:
    sys.path.insert(0, str(VLMEVALKIT_ROOT))
    import run as vrun  # the vendored stock run.py

    limit = int(os.environ.get("VLMEVAL_SAMPLE_LIMIT", "0") or 0)
    seed = int(os.environ.get("VLMEVAL_SAMPLE_SEED", "0") or 0)
    if limit > 0:
        orig = vrun.build_dataset_from_config

        def build_limited(cfg, dataset_name):
            ds = orig(cfg, dataset_name)
            if ds is not None:
                truncate_dataset(ds, limit, seed)
            return ds

        vrun.build_dataset_from_config = build_limited

    sys.argv = [str(VLMEVALKIT_ROOT / "run.py")] + sys.argv[1:]
    vrun.load_env()
    vrun.main()


if __name__ == "__main__":
    main()
