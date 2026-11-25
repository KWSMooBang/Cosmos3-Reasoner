# Cosmos3-Reasoner evaluation

Evaluates NVIDIA Cosmos 3 Reasoner (`nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super`) and Qwen3-VL
baselines on 3D spatial/perception and physics benchmarks, using NVIDIA's modified VLMEvalKit
(vendored in `third_party/vlmevalkit-cosmos`, see `third_party/UPSTREAM.md`).

Evaluation settings (sampling, thinking, frames, judges, pinned versions) are in **PROTOCOL.md**.
Known gaps (stub benchmarks, spec-vs-code differences) are in **TODO.md**.

```
configs/models/   model layer   (one JSON per model x thinking x sampling variant)
configs/data/     dataset layer (spatial/<bench>.json, physics/<bench>.json; frame settings, provenance)
configs/suites/   smoke.yaml, spatial.yaml, physics.yaml (bench list + run.py flags + sample limits)
scripts/          run_suite.py, collect.py, check_suites.py, vlmeval_run.py (sample-limit wrapper)
serving/          vLLM server scripts (separate task; not part of this scaffold)
results/          summary.csv is committed; results/raw/ (predictions, logs) is gitignored
```

## Quick start

### 1. Install the eval client (CPU only; separate from the vLLM server venv)

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
cd third_party/vlmevalkit-cosmos
uv pip install -r requirements.txt
uv pip install einops accelerate "setuptools<81" evaluate
uv pip install -e .
cd ../..
python -c "import vlmeval; print('vlmeval OK')"

cp .env.example .env                       # then edit: endpoint, LMUData, judge key
mkdir -p /data0/Cosmos3-Reasoner/LMUData   # LMUData must exist (else VLMEvalKit uses ~/LMUData)
```

`nvdataset` (NVIDIA-internal DSS client, only for TAR/AETCBench) is not installed and not needed.

### 2. Serve the model (native vLLM)

Serving uses a natively installed vLLM in its own venv (`.venv-vllm`); see `serving/`:

```bash
serving/setup_vllm_env.sh          # one-time: creates .venv-vllm
serving/serve_cosmos3_nano.sh      # OpenAI-compatible server on :8080
```

The server must expose `POST /v1/chat/completions` and `GET /v1/models`, accept local media and
honour per-request `mm_processor_kwargs` (NVIDIA's flags: `--allowed-local-media-path /
--mm-processor-cache-gb 0 --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}'`,
`--max-model-len 128000`). NVIDIA's reference environment is the docker image
`vllm/vllm-openai:cosmos3@sha256:db0bb920…` (PROTOCOL.md §1); docker is not available on this
machine, so it serves only as a reference.

### 3. Run

```bash
python scripts/check_suites.py                                         # validate suites (offline)
python scripts/check_video_sampling.py                                 # prove frame counts per video bench (offline)
python scripts/check_revsi_frames.py                                   # ReVSI: official frames reach the request (needs data)
python scripts/test_revsi_scoring.py                                   # ReVSI scorer tests (offline, needs parquet)
python scripts/check_spacecast_parity.py                               # SpaceCast-Bench vs official evaluator (offline, ~6 min; needs the built cache)
python scripts/test_physbench_scoring.py                               # PhysBench scorer vs official + lmms-eval (offline after first label fetch)
python scripts/check_physbench.py                                      # PhysBench: order, frames/video, image cap, token counts (needs data; --key PhysBench_native)
python scripts/test_physbench_media.py                                 # PhysBench media manifest repair test (needs data)
python scripts/check_vstibench_frames.py                              # VSTI-Bench: 32 frames reach the request (needs data)
python scripts/test_vstibench_scoring.py --official-dir <VLM-3R>/thinking-in-space/lmms_eval/tasks/vstibench   # VSTI-Bench vs official scorer
python scripts/test_3dsrbench_official.py --official-script <compute_3drbench_results_circular.py>             # 3DSRBench_official vs authors' scorer (~3 min)
python scripts/test_quantiphy_scoring.py                               # QuantiPhy prompt/parser/MRA vs official code (offline; needs data, fetches official code once)
python scripts/check_quantiphy_frames.py                               # QuantiPhy: every frame reaches the request (needs data)
.venv-vllm/bin/python scripts/quantiphy_token_budget.py                # QuantiPhy prompt tokens with the Cosmos3-Nano processor (after check_quantiphy_frames)
python scripts/run_suite.py --suite configs/suites/smoke.yaml --dry-run # print merged configs, no server
python scripts/run_suite.py --suite configs/suites/smoke.yaml           # cosmos3_nano_nothink by default
python scripts/run_suite.py --suite configs/suites/spatial.yaml --model cosmos3_nano_think
python scripts/run_suite.py --suite configs/suites/physics.yaml --model qwen3vl_8b --benchmarks SeePhys
```

SpaceCast-Bench runs in the official **cot** mode by default (`spatial/SpaceCastBench_CoT`, the
official main experiment); the direct mode `spatial/SpaceCastBench` is a commented opt-in line in
`spatial.yaml` (PROTOCOL.md §6, SpaceCast-Bench).

PhysBench (`physics/PhysBench`, 10,002 interleaved image/video MCQ, official 8 frames per video and the
official prefix-match scorer, comparable to the leaderboard; images above 6.9 MP downsized so the largest prompt
fits TP=1 / 16384) is in the physics suite; `physics/PhysBench_val` (200, answer-format pilot) and
`physics/PhysBench_native` (native resolution, TP=2 / 65536) are commented opt-in lines (PROTOCOL.md §6, PhysBench).

The spatial suite runs `spatial/VSTIBench` (VLM-3R VSTI-Bench, 32 frames, headline = paper Avg) and
`spatial/3DSRBench_official` (official CircularEval) instead of the upstream `3DSRBench` key, whose 4-setting
DataFrame parse_score reads as 0.00 (PROTOCOL.md §6, VSTI-Bench / 3DSRBench).

QuantiPhy (`physics/QuantiPhy`) is scored on the official 158-question validation split, the only split with
public ground truth. Compare it with the project page's validation leaderboard, not with paper Table 1, which
covers the whole set. `physics/QuantiPhy_test` (3,289 questions, GT withheld) only writes the official
submission CSV (PROTOCOL.md §6, QuantiPhy).

Per bench, `run_suite.py` composes (same rule as `cosmos_eval/run_all.py`)

```
model_conf   = {class} | model.defaults | data.video_sampling | model.benchmarks[bench]
dataset_conf = data[bench] | {model_family}
```

writes `results/raw/<model>/<suite>/_configs/<bench>__<mode>.json`, runs the stock
`run.py --config <json> --verbose --save-eval-results` (cwd = vendored kit) with work dir
`results/raw/<model>/<suite>/<bench>/<mode>/`, where `<mode>` is `full` or
`limit<N>_seed<S>`, and stores the `parse_score.py` headline score of **the eval JSON written by
that run** (exact file name, mtime after launch; older files are ignored) in
`<bench>/<mode>/score.json` (`run.log` next to it). Useful flags: `--benchmarks A,B`,
`--sample-limit N` (0 = full), `--sample-seed S`, `--concurrency N`, `--work-dir DIR`,
`--api-nproc N` (parallel requests), `--reuse` (resume an interrupted run: without it run.py moves the
same-day partial predictions to `bak_*/` and restarts from zero),
`--model <name|path.json>`.

Model variants (`configs/models/`): `cosmos3_{nano,super}_nothink` (**reproduction default**:
NVIDIA's `cosmos_eval/models/cosmos.json` settings, T=0, thinking off), `_think` (official
`<think>` prompt, T=0), `_think_nvprompt` (NVIDIA's upstream `CosmosReason2Think` prompt),
`_nothink_official` / `_think_official` (official sampling), `qwen3vl_{8b,32b}`.

### 4. Collect

```bash
python scripts/collect.py      # results/raw/**/score.json -> results/summary.csv
```

## Sample limit (smoke runs)

Stock VLMEvalKit `run.py` has no sample-limit option. When a suite entry has `sample_limit: N`
(or `--sample-limit N`), `run_suite.py` launches `scripts/vlmeval_run.py` instead of `run.py`
with the same arguments plus `VLMEVAL_SAMPLE_LIMIT=N VLMEVAL_SAMPLE_SEED=S` (`sample_seed`,
default 0; both are cleared from the environment of full runs). The
wrapper imports the unmodified `run.py`, and right after `build_dataset_from_config` builds the
dataset it replaces `dataset.data` with a seeded random subset of N rows (original order kept;
`dataset.videos` filtered to match). Everything else, including evaluation, is stock. The full
TSV/videos are still downloaded on first use (the dataset constructor fetches them before the
subset is taken). Smoke scores are over the subset only and must never be reported as benchmark
numbers. Subset and full runs never share a run.py work dir (`<bench>/full/` vs
`<bench>/limit<N>_seed<S>/`; run.py would otherwise reuse the same `T<date>_G<commit>/`
prediction dir for every run of the day), and `summary.csv` keeps one row per
(model, suite, bench, mode) with `mode`, `sample_limit` and `sample_seed` columns.
