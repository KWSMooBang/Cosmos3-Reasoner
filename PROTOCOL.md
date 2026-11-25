# Evaluation protocol

Status: draft (2026-11-25). Nothing in this file has been validated by an actual run yet.
Every value says where it comes from. "NVIDIA kit" = `third_party/vlmevalkit-cosmos` (NVIDIA
cosmos-framework `.agents/skills/evaluation/cosmos3/reasoner/vlmevalkit` @ `a3fb346`).
"Official guide" = NVIDIA/cosmos `cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md` @ `1fe7e3be`
(mirrored verbatim in cosmos-framework `.agents/skills/cosmos3-inference-reasoning/references/prompt-guide.md`).

## 1. Serving

| item | value | source |
|---|---|---|
| server used here | **native vLLM `<version from t002>`** in `.venv-vllm` (`serving/`) | human decision 2026-11-25: no docker on this host |
| reference environment | `vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55` (vLLM `0.23.1rc1.dev1306+gbfee6a802`) | NVIDIA kit README §2 |
| server flags (reference) | `VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve <model> --async-scheduling --allowed-local-media-path / --port 8080 --max-model-len 128000 --gpu-memory-utilization 0.9 --enable-chunked-prefill --mm-processor-cache-gb 0 --mm-encoder-tp-mode data --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}'` | NVIDIA kit README §2 |
| reasoning parser | none (NVIDIA's command has none; the client parses `<think>` itself) | NVIDIA kit README §2 |

Any deviation of the native server from the reference flags (e.g. TP size, max-model-len for
24 GB GPUs) must be recorded here by t002 / the run that uses it.

## 2. Models

| model | HF revision | verified |
|---|---|---|
| `nvidia/Cosmos3-Nano` | `45666784803b3a7ad577e0def405f9afcc3bddb2` (lastModified 2026-11-12T23:42:25Z) | HF API, 2026-11-25 |
| `nvidia/Cosmos3-Super` | `73d5746645961a0ec9607123a003bd9c1f8a0282` (lastModified 2026-11-12T23:42:22Z) | HF API, 2026-11-25 |
| `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-32B-Instruct` | to be pinned when first served | — |

Note: the NVIDIA kit README derives Nano's processor profile from revision `e59a53c2…`, not from
the revision above; patch 16 / merge 2 / temporal patch 2 should be re-checked against `45666784`.

The Cosmos3 chat template (`chat_template.json` @ `45666784`) has **no `enable_thinking`
switch** and inserts no default system prompt; `generation_config.json` has
`temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0` (vLLM uses these as defaults
for any sampling field the request leaves `null`).

## 3. Client and sampling

Client: NVIDIA kit `run.py --config … --verbose --save-eval-results` via `scripts/run_suite.py`,
API wrapper `vlmeval/api/cosmos_reason.py` (`CosmosReason2`, image patch 16).
`run.py` flags: `--api-nproc 8 --judge-nproc 4` (NVIDIA `cosmos_eval/manifest.json`).

| model config | class | T | top_p | top_k | presence | rep. | system prompt | max_tokens | source |
|---|---|---|---|---|---|---|---|---|---|
| `*_nothink` (**reproduction default**) | CosmosReason2 | 0 | null | null | null | null | none | 8192 | NVIDIA `cosmos_eval/models/cosmos.json` defaults (also `seed 1, retry 10, timeout 600, chat_template_kwargs.enable_thinking=false`) |
| `*_think` | CosmosReason3Think | 0 | null | null | null | null | "You are a helpful assistant." | 16384 | official guide prompt; greedy like the default |
| `*_think_nvprompt` | CosmosReason2Think | 0 | null | null | null | null | none | 16384 | NVIDIA kit upstream thinking class |
| `*_nothink_official` | CosmosReason2 | 0.7 | 0.8 | 20 | 1.5 | 1.0 | "You are a helpful assistant." | 8192 | official guide "without reasoning" |
| `*_think_official` | CosmosReason3Think | 0.6 | 0.95 | 20 | 0.0 | 1.0 | "You are a helpful assistant." | 16384 | official guide "with reasoning" |
| `qwen3vl_{8b,32b}` | CosmosReason2 | 0 | null | null | null | null | none | 8192 | same as reproduction default (NVIDIA `vlmeval/config.py` drives Qwen3-VL with CosmosReason2) |

- `max_tokens 16384` for thinking variants is our choice (8192 risks truncating the trace); the
  wrapper logs `finish_reason != stop` as "truncated" and then scores the raw (unparsed) text.
- `enable_thinking=false` is inherited from NVIDIA's defaults but is a no-op for the Cosmos3
  template; thinking variants send no `chat_template_kwargs`.
- Seed: `seed=1` on every request (NVIDIA default). With T>0 the result is only reproducible
  for the same server build, batch composition aside. `THINKING_RETRY=1` (no resampling).

## 4. Thinking on/off

Official guide: reasoning is switched **only through the user prompt**. With reasoning, append
after the task text:

```
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```

system prompt `You are a helpful assistant.`; media before text; the cookbook parser keeps
`text.split("</think>")[-1]`.

Check of NVIDIA kit `CosmosReason2Think` (`vlmeval/api/cosmos_reason.py`): **does not match**.
It appends `\nPlease provide your detailed reasoning within <think> and </think> tags before
giving the final answer.` to *every* text item of every user turn, sends no system prompt, and
parses with `re.match(r"^<think>(.*)<\/think>(.+)$", DOTALL)`, which returns the whole raw
reply (reasoning included) when the reply does not begin with `<think>`.
`CosmosReason2.parse_answer` (non-thinking) strips a leading `<think>…</think>` block if present.

Added `CosmosReason3Think` (local edit, separate commit; see `third_party/UPSTREAM.md`): official
system prompt, official instruction appended once after the last text item of the final user turn,
answer = text after the last `</think>` (raw reply if no `</think>`). `CosmosReason2Think`
remains available as `*_think_nvprompt` for comparison.

## 5. Prompts and media

- MCQ datasets use the dataset's own VLMEvalKit prompt (`CosmosReason.use_custom_prompt` returns
  False for MCQ). VQA-type image datasets not in the wrapper's exclusion list (e.g. PhyX, SeePhys,
  QSpatial, Spatial457, CoreCognition, ERQA) get the wrapper's prompt, which appends
  `\nAnswer the question using a single word or phrase.` (Omni3D: question only; SparBench,
  RefSpatial, RoboSpatialHome, EmbSpatialBench: dataset prompt). This is NVIDIA's behaviour; kept.
- Images: base64 at native resolution (`img_size=-1`); resizing is done server-side by the
  checkpoint's processor (`preprocessor_config.json`).
- Videos: CosmosReason* are `VIDEO_LLM=True`, so video datasets pass the raw video path and the
  **client** samples frames with qwen-vl-utils 0.0.14 (`process_vision_info`), re-encodes them as a
  base64 mp4 and sends `mm_processor_kwargs={fps, do_sample_frames: false}`. The dataset-side
  `nframe`/`fps` are *not* used on this path, so each video bench's data config has a
  `video_sampling` block that `run_suite.py` merges into the model config (the dataset-side
  values are set to the same numbers for consistency). Unset resolution keys fall back to
  qwen-vl-utils defaults: per-frame max `768 x 32^2 = 786,432` px, min `128 x 32^2` px,
  `total_pixels = 128000 x 32^2 x 0.9`.

## 6. Per-benchmark settings

Frames: `uniform` = `np.linspace(0, n-1, k)` (VSI-Bench paper / lmms-eval convention; the
client's qwen-vl-utils `nframes` sampling is also uniform linspace).

| bench | suite | modality | frames | scoring | judge | frame source |
|---|---|---|---|---|---|---|
| VSI-Bench | spatial, smoke | video | 32 uniform | MCQ rule + NA (MRA) | exact_matching | task spec; VSI-Bench paper; kit variant `VSI-Bench_32frame` |
| VSI-Bench-Debiased | spatial | video | 32 uniform | as VSI-Bench | exact_matching | same as VSI-Bench |
| MMSIVideoBench | spatial | video | 64 | MCQ rule | exact_matching | kit variant `_64frame`; paper SC policy = 300 frames (TODO) |
| STI-Bench | spatial | video | 30 | MCQ rule | exact_matching | official paper setting (kit comment) |
| SiteBenchVideo | spatial | video | 32 | MCQ rule | exact_matching | kit variant `_32frame` (no official value annotated) |
| DSRBench | spatial | video | 1 fps | MCQ rule | exact_matching | official paper setting (kit comment) |
| MMSIBench_wo_circular, OmniSpatialBench, ViewSpatialBench, MindCubeBench_raw_qa (smoke: _tiny), SiteBenchImage, EmbSpatialBench, ERQA, 3DSRBench, BLINK, MMVP | spatial | image | – | MCQ rule extraction | exact_matching | – |
| SparBench | spatial | image | – | MCQ rule + NA | exact_matching (rule path) | – |
| Spatial457, Omni3DBench | spatial | image | – | dataset-internal rule metric | none used | – |
| RefSpatial, RoboSpatialHome | spatial | image | – | Point2D in-mask | none used | – |
| QSpatial_plus / _scannet | spatial | image | – | rule extraction (no judge passed) | none (a judge would switch it to LLM extraction) | – |
| CoreCognition | spatial | image | – | hybrid matcher, exact-match fallback | exact_matching (forces fallback; NVIDIA run.py default would be gpt-4.1) | – |
| PhyX_mini_MC, PhyX_MC | physics | image | – | string-level (`valid_type=STR`) | none | – |
| SeePhys | physics | image | – | LLM extraction + compare | **`gpt-4o-mini-2024-07-18`** (OpenAI, pinned snapshot) | – |

Judge policy: rule-based/exact matching wherever the dataset supports it (all MCQ benches,
VSI/Spar NA questions, PhyX). The only LLM-judged bench is SeePhys (its evaluate() requires one;
VLMEvalKit default would be `deepseek` via SiliconFlow). Pinned judge: OpenAI
`gpt-4o-mini-2024-07-18`, judge `max_tokens` as set by the dataset, `retry 3`, `--judge-nproc 4`.
MCQ benches pass `--judge exact_matching`, overriding run.py's default `chatgpt-0125` fallback
extractor.

## 7. Reproducibility caveats

- QSpatial downloads its prompt templates from the Q-Spatial-Bench GitHub `main` branch at run time (not pinned).
- Dataset TSVs are verified by the MD5 in each dataset class where one is given; HF-hosted
  datasets (Omni3DBench, CoreCognition, PhyX, SeePhys, Q-Spatial) are fetched from `main`.
- Smoke runs (`sample_limit`) score a seeded random subset (seed 0); never report them.
