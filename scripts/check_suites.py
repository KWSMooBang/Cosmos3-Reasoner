#!/usr/bin/env python
"""Validate suites/configs against the vendored VLMEvalKit (no server, no data download).

For every bench in configs/suites/*.yaml checks that
  * the bench key is in vlmeval.dataset.SUPPORTED_DATASETS,
  * configs/data/<domain>/<bench>.json exists and its `class` is a vlmeval.dataset class that
    lists the key (or its `dataset` field) in supported_datasets(),
  * that class's module is not a public-release stub
    ("dataset implementation not included in this public release"),
  * every configs/models/*.json has the keys run_all.compose needs.

    python scripts/check_suites.py                 # validate all suites
    python scripts/check_suites.py --list-stubs    # also print stub modules / classes
    python scripts/check_suites.py --owner VSI-Bench SparBench   # which class serves a key
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import warnings
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
STUB_MARK = "not included in this public release"


def load_vlmeval():
    os.environ.setdefault("LMUData", str(REPO / ".lmudata_unused"))  # import-time only, nothing is fetched
    warnings.filterwarnings("ignore")
    import vlmeval.dataset as D
    return D


def is_stub_class(cls) -> bool:
    mod = sys.modules.get(cls.__module__)
    return bool(mod and STUB_MARK in (mod.__doc__ or ""))


def owners(D, key: str) -> list:
    out = []
    for cls in D.DATASET_CLASSES:
        try:
            if key in cls.supported_datasets():
                out.append(cls)
        except Exception:
            pass
    return out


def stub_modules(D) -> list[tuple[str, list[str]]]:
    res = []
    for name, mod in sorted(sys.modules.items()):
        if name.startswith("vlmeval.dataset") and STUB_MARK in (getattr(mod, "__doc__", "") or ""):
            classes = [n for n, o in vars(mod).items() if inspect.isclass(o) and o.__module__ == name]
            res.append((name, classes))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-stubs", action="store_true")
    ap.add_argument("--owner", nargs="*", help="print the class/module serving these keys and exit")
    args = ap.parse_args(argv)

    D = load_vlmeval()
    supported = set(D.SUPPORTED_DATASETS)

    if args.owner:
        for key in args.owner:
            os_ = owners(D, key)
            print(f"{key}: in SUPPORTED_DATASETS={key in supported}; "
                  + ", ".join(f"{c.__name__} ({c.__module__}{', STUB' if is_stub_class(c) else ''})" for c in os_))
        return

    if args.list_stubs:
        print("== stub modules (public-release placeholders) ==")
        for name, classes in stub_modules(D):
            print(f"  {name}: {', '.join(classes)}")

    errors = []
    for mpath in sorted((CONFIGS / "models").glob("*.json")):
        m = json.loads(mpath.read_text())
        for k in ("model_family", "model_key", "class", "defaults"):
            if k not in m:
                errors.append(f"{mpath.name}: missing '{k}'")
        import vlmeval.api as A
        if not hasattr(A, m.get("class", "")):
            errors.append(f"{mpath.name}: class {m.get('class')} not in vlmeval.api")

    n = 0
    for spath in sorted((CONFIGS / "suites").glob("*.yaml")):
        suite = yaml.safe_load(spath.read_text())
        for entry in suite.get("benchmarks", []):
            key = entry if isinstance(entry, str) else entry["key"]
            domain, bench = key.split("/", 1)
            n += 1
            where = f"{spath.name}:{key}"
            if bench not in supported:
                errors.append(f"{where}: '{bench}' not in SUPPORTED_DATASETS")
            dpath = CONFIGS / "data" / domain / f"{bench}.json"
            if not dpath.exists():
                errors.append(f"{where}: missing {dpath.relative_to(REPO)}")
                continue
            dconf = json.loads(dpath.read_text())
            cls = getattr(D, dconf.get("class", ""), None)
            if cls is None:
                errors.append(f"{where}: class {dconf.get('class')} not in vlmeval.dataset")
                continue
            if is_stub_class(cls):
                errors.append(f"{where}: class {cls.__name__} is a public-release stub")
            ds_name = dconf.get("dataset", bench)
            try:
                ok = ds_name in cls.supported_datasets()
            except Exception as e:  # pragma: no cover
                ok, _ = False, e
            if not ok:
                errors.append(f"{where}: {cls.__name__}.supported_datasets() lacks '{ds_name}'")
            for o in owners(D, bench):
                if is_stub_class(o):
                    errors.append(f"{where}: key also claimed by stub {o.__name__}")

    if errors:
        print("\n".join(["FAIL:"] + [f"  {e}" for e in errors]))
        raise SystemExit(1)
    print(f"OK: {n} suite entries valid (in SUPPORTED_DATASETS, non-stub class, data config present).")


if __name__ == "__main__":
    main()
