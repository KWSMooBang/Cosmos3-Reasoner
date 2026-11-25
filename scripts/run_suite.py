#!/usr/bin/env python
"""Run a benchmark suite against one model layer with the vendored NVIDIA VLMEvalKit.

Mirrors third_party/vlmevalkit-cosmos/cosmos_eval/run_all.py (and reuses its compose / env
rendering / command builder and parse_score.py), but reads our own config layers:

    configs/models/<model>.json        model layer  (same schema as cosmos_eval/models/cosmos.json)
    configs/data/<domain>/<bench>.json dataset layer (same schema as cosmos_eval/data/*/*.json)
    configs/suites/<suite>.yaml        which benches to run + run.py flags (api_nproc, ...)

Per benchmark:
    model_conf   = {class} | model.defaults | model.benchmarks[bench]
    dataset_conf = data[bench] | {model_family}
    -> <work-dir>/_configs/<bench>__<mode>.json
    -> run.py --config <json> --work-dir <work-dir>/<bench>/<mode> --api-nproc N --judge-nproc M
              [--judge J] --verbose --save-eval-results
    -> parse_score.report(<the eval JSON this run wrote>) -> <work-dir>/<bench>/<mode>/score.json
    <mode> = full | limit<N>_seed<S>   (subset and full runs never share a run.py work dir)

Examples:
    python scripts/run_suite.py --suite configs/suites/smoke.yaml --dry-run
    python scripts/run_suite.py --suite configs/suites/spatial.yaml --model cosmos3_nano_nothink
    python scripts/run_suite.py --suite configs/suites/smoke.yaml --benchmarks VSI-Bench_32frame
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
VLMEVALKIT_ROOT = REPO / "third_party" / "vlmevalkit-cosmos"
COSMOS_EVAL = VLMEVALKIT_ROOT / "cosmos_eval"
LIMITED_RUN_PY = REPO / "scripts" / "vlmeval_run.py"

sys.path.insert(0, str(COSMOS_EVAL))
import parse_score  # noqa: E402  (vendored, stdlib only)
import run_all  # noqa: E402  (vendored, stdlib only)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def load_model_layer(model: str) -> dict[str, Any]:
    path = Path(model) if model.endswith(".json") else CONFIGS / "models" / f"{model}.json"
    if not path.exists():
        have = ", ".join(p.stem for p in sorted((CONFIGS / "models").glob("*.json")))
        sys.exit(f"Error: no model layer at {path} (have: {have})")
    layer = json.loads(path.read_text("utf-8"))
    layer.setdefault("benchmarks", {})
    return layer


def load_data_conf(domain: str, bench: str) -> dict[str, Any]:
    path = CONFIGS / "data" / domain / f"{bench}.json"
    if not path.exists():
        sys.exit(f"Error: no dataset layer at {path}")
    conf = json.loads(path.read_text("utf-8"))
    # Keys starting with "_" are documentation (provenance etc.), not run.py arguments.
    return {k: v for k, v in conf.items() if not k.startswith("_")}


def load_suite(path: Path) -> dict[str, Any]:
    suite = yaml.safe_load(path.read_text("utf-8"))
    suite.setdefault("name", path.stem)
    suite.setdefault("run", {})
    suite.setdefault("benchmarks", [])
    return suite


def select_benches(suite: dict[str, Any], only: list[str] | None) -> list[dict[str, Any]]:
    """Suite entries -> [{key, domain, bench, run, sample_limit, sample_seed, idx}].

    An entry is either "domain/bench" or a mapping {key: "domain/bench", <run overrides>,
    sample_limit: N, sample_seed: S}. Run flags: suite.run | entry overrides.
    """
    out = []
    for entry in suite["benchmarks"]:
        if isinstance(entry, str):
            entry = {"key": entry}
        entry = dict(entry)
        key = entry.pop("key")
        domain, bench = key.split("/", 1)
        if only and bench not in only:
            continue
        sample_limit = entry.pop("sample_limit", suite.get("sample_limit"))
        sample_seed = entry.pop("sample_seed", suite.get("sample_seed", 0))
        run_flags = {**suite["run"], **entry}
        out.append({"key": key, "domain": domain, "bench": bench, "run": run_flags,
                    "sample_limit": sample_limit or None, "sample_seed": int(sample_seed)})
    for i, it in enumerate(out):
        it["idx"] = i
    return out


VIDEO_SAMPLING_KEYS = ("nframes", "fps", "max_pixels", "total_pixels", "min_pixels", "max_frames")


def build_config(model_layer: dict[str, Any], item: dict[str, Any], *, render: bool) -> dict[str, Any]:
    """Full `run.py --config` document.

    Composition is cosmos_eval.run_all.compose verbatim, plus one extension: a dataset layer may
    carry `video_sampling` (nframes/fps/max_pixels/total_pixels/...). CosmosReason* clients are
    VIDEO_LLM=True, so video benches hand them the raw video and the *client* samples frames from
    its own model-level attributes; the dataset's `nframe`/`fps` are not used on that path. The
    block is therefore merged into model_conf, below the model layer's per-bench override:
        model_conf = {class} | defaults | data.video_sampling | model.benchmarks[bench]
    """
    bench = item["bench"]
    data_conf = load_data_conf(item["domain"], bench)
    video_sampling = data_conf.pop("video_sampling", {})
    unknown = set(video_sampling) - set(VIDEO_SAMPLING_KEYS)
    if unknown:
        sys.exit(f"Error: {item['key']}: unknown video_sampling keys {sorted(unknown)}")
    model_conf, dataset_conf = run_all.compose(data_conf, model_layer, bench)
    per_bench = model_layer["benchmarks"].get(bench, {})
    for k, v in video_sampling.items():
        if k not in per_bench:
            model_conf[k] = v
    # qwen-vl-utils (client) and VideoBaseDataset (dataset) each accept only one of the two.
    if model_conf.get("nframes") is not None and model_conf.get("fps") is not None:
        sys.exit(f"Error: {item['key']}: model config sets both nframes={model_conf['nframes']} and "
                 f"fps={model_conf['fps']}; set exactly one (data video_sampling / model per-bench override)")
    if (dataset_conf.get("nframe") or 0) > 0 and (dataset_conf.get("fps") or -1) > 0:
        sys.exit(f"Error: {item['key']}: dataset config sets both nframe and fps; set exactly one")
    if render:
        model_conf = run_all._render_env(model_conf)
        dataset_conf = run_all._render_env(dataset_conf)
    return {"model": {model_layer["model_key"]: model_conf}, "data": {item["bench"]: dataset_conf}}


def run_mode(item: dict[str, Any]) -> str:
    """Artifact sub-dir for this run: 'full' or 'limit<N>_seed<S>'.

    Subset and full runs must never share a run.py work dir: run.py reuses the same
    <work-dir>/<model>/T<date>_G<commit>/ for every run of the day, and an eval-file finder
    would otherwise pick up another mode's eval JSON.
    """
    if item.get("sample_limit"):
        return f"limit{item['sample_limit']}_seed{item['sample_seed']}"
    return "full"


def bench_dir(work_root: Path, item: dict[str, Any]) -> Path:
    # run_all._bench_workdir keeps ConcatDataset benches name-free; the mode dir is name-free too.
    return run_all._bench_workdir(work_root, item) / run_mode(item)


def config_path_for(configs_dir: Path, item: dict[str, Any]) -> Path:
    return configs_dir / f"{item['bench']}__{run_mode(item)}.json"


def find_own_eval_json(bench_out: Path, model_key: str, bench: str, t0: float) -> Path | None:
    """The eval JSON written by *this* run.py invocation (exact file name, mtime >= launch time)."""
    cands = [p for suffix in ("dict", "df")
             for p in bench_out.glob(f"{model_key}/*/{model_key}_{bench}.{suffix}.eval.json")
             if p.stat().st_mtime >= t0]
    return max(cands, key=lambda p: p.stat().st_mtime, default=None)


def run_cmd(config_path: Path, work_dir: Path, item: dict[str, Any]) -> list[str]:
    cmd = run_all._run_cmd(config_path, work_dir, item["run"])
    if item.get("sample_limit"):
        # Same argv, but go through the sample-limit wrapper instead of stock run.py.
        cmd[1] = str(LIMITED_RUN_PY)
    if item["run"].get("reuse"):
        # Without --reuse, run.py moves this day's partial predictions (*_supp.pkl) of the same
        # work dir into bak_*/ on a re-run, i.e. an interrupted run restarts from zero.
        cmd.append("--reuse")
    return cmd


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_one(item: dict[str, Any], model_layer: dict[str, Any], *, model_name: str, suite_name: str,
            work_root: Path, configs_dir: Path) -> dict[str, Any]:
    bench = item["bench"]
    bench_out = bench_dir(work_root, item)
    bench_out.mkdir(parents=True, exist_ok=True)

    cfg = build_config(model_layer, item, render=True)
    config_path = config_path_for(configs_dir, item)
    config_path.write_text(json.dumps(cfg, indent=2), "utf-8")

    env = os.environ.copy()
    env.pop("VLMEVAL_SAMPLE_LIMIT", None)
    env.pop("VLMEVAL_SAMPLE_SEED", None)
    if item.get("sample_limit"):
        env["VLMEVAL_SAMPLE_LIMIT"] = str(item["sample_limit"])
        env["VLMEVAL_SAMPLE_SEED"] = str(item["sample_seed"])
    cmd = run_cmd(config_path, bench_out, item)
    log_path = bench_out / "run.log"
    started = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.time() - 1.0  # mtime granularity slack
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=str(VLMEVALKIT_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)

    row = {"model": model_name, "suite": suite_name, "bench": bench, "domain": item["domain"],
           "mode": run_mode(item), "sample_limit": item.get("sample_limit") or "",
           "sample_seed": item["sample_seed"] if item.get("sample_limit") else "",
           "started": started, "finished": dt.datetime.now().isoformat(timespec="seconds"),
           "config": str(config_path), "log": str(log_path)}
    if proc.returncode != 0:
        row.update(status="error", score=None, detail=f"run.py exit {proc.returncode}")
    else:
        own = find_own_eval_json(bench_out, model_layer["model_key"], bench, t0)
        rep = (parse_score.report(eval_json=str(own), dataset_name=bench) if own is not None
               else {"eval_json": None})
        if rep["eval_json"] is None:
            row.update(status="no-eval", score=None,
                       detail="this run wrote no *.eval.json (see run.log); older files are ignored")
        else:
            row.update(status="ok", score=rep["overall"], subscores=rep["subscores"],
                       eval_json=rep["eval_json"])
            # Benches with a diagnostic lenient-parser score (ReVSI: Overall_lenient, PROTOCOL §6.1).
            # The headline `score` stays parse_score's Overall (official parser for ReVSI).
            if isinstance(rep["subscores"].get("Overall_lenient"), (int, float)):
                row["score_lenient"] = rep["subscores"]["Overall_lenient"]
    (bench_out / "score.json").write_text(json.dumps(row, indent=2), "utf-8")
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, help="configs/suites/<name>.yaml")
    ap.add_argument("--model", default=None,
                    help="model layer name in configs/models/ or a .json path (default: suite's `model`)")
    ap.add_argument("--benchmarks", default=None, help="comma-separated subset of the suite's bench names")
    ap.add_argument("--sample-limit", type=int, default=None,
                    help="override every bench's sample_limit (0 = full benchmark)")
    ap.add_argument("--sample-seed", type=int, default=None, help="override every bench's sample_seed")
    ap.add_argument("--concurrency", type=int, default=None, help="benches run in parallel (default: suite or 1)")
    ap.add_argument("--work-dir", default=None,
                    help="output root (default: results/raw/<model>/<suite>)")
    ap.add_argument("--api-nproc", type=int, default=None,
                    help="override every bench's run.api_nproc (parallel requests to the server)")
    ap.add_argument("--reuse", action="store_true",
                    help="pass --reuse to run.py: a re-run resumes from this work dir's finished/partial "
                         "predictions instead of moving them to bak_*/ (full and limit runs never share a dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the merged per-bench config JSON and run.py command; contact no server")
    args = ap.parse_args(argv)

    load_dotenv(REPO / ".env")
    suite = load_suite(Path(args.suite))
    model_name = args.model or suite.get("model") or "cosmos3_nano_nothink"
    model_layer = load_model_layer(model_name)
    model_name = Path(model_name).stem
    items = select_benches(suite, run_all._csv(args.benchmarks))
    if args.sample_limit is not None:
        for it in items:
            it["sample_limit"] = args.sample_limit or None
    if args.sample_seed is not None:
        for it in items:
            it["sample_seed"] = args.sample_seed
    for it in items:
        if args.api_nproc is not None:
            it["run"]["api_nproc"] = args.api_nproc
        if args.reuse:
            it["run"]["reuse"] = True
    if not items:
        sys.exit("Error: no benchmarks selected.")

    work_root = Path(args.work_dir or REPO / "results" / "raw" / model_name / suite["name"]).resolve()
    configs_dir = work_root / "_configs"

    if args.dry_run:
        for it in items:
            cfg = build_config(model_layer, it, render=True)
            cmd = run_cmd(config_path_for(configs_dir, it), bench_dir(work_root, it), it)
            env = (f"VLMEVAL_SAMPLE_LIMIT={it['sample_limit']} VLMEVAL_SAMPLE_SEED={it['sample_seed']} "
                   if it.get("sample_limit") else "")
            print(f"### {it['key']}  (model={model_name}, mode={run_mode(it)})")
            print(json.dumps(cfg, indent=2))
            print(f"$ (cd {VLMEVALKIT_ROOT} && {env}{' '.join(cmd)})\n")
        print(f"[dry-run] {len(items)} benchmark(s); nothing was executed.")
        return

    if not os.environ.get("COSMOS_API_BASE"):
        print("Warning: COSMOS_API_BASE is not set (see .env.example)", file=sys.stderr)
    # vlmeval.smp.LMUDataRoot() silently falls back to ~/LMUData when $LMUData is unset OR missing.
    lmu = os.environ.get("LMUData")
    if not lmu or not Path(lmu).is_dir():
        sys.exit(f"Error: LMUData={lmu!r} is not an existing directory; VLMEvalKit would silently use "
                 f"~/LMUData instead. Create it (e.g. mkdir -p /data0/Cosmos3-Reasoner/LMUData) or fix .env.")

    configs_dir.mkdir(parents=True, exist_ok=True)
    conc = args.concurrency or suite.get("concurrency", 1)
    print(f"Running {len(items)} benchmark(s) [{suite['name']}] on {model_name}, concurrency {conc} -> {work_root}")
    results = []
    with ThreadPoolExecutor(max_workers=max(1, conc)) as pool:
        futs = {pool.submit(run_one, it, model_layer, model_name=model_name, suite_name=suite["name"],
                            work_root=work_root, configs_dir=configs_dir): it for it in items}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            score = f"{r['score']:.2f}" if isinstance(r.get("score"), (int, float)) else "-"
            if isinstance(r.get("score_lenient"), (int, float)):
                score += f" (lenient {r['score_lenient']:.2f})"
            print(f"  [{r['status']}] {r['bench']}: {score}  {r.get('detail', '')}")
    ok = sum(r["status"] == "ok" for r in results)
    print(f"\n{ok}/{len(results)} ok. Collect with: python scripts/collect.py")
    if ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
