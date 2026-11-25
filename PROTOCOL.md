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

- `max_tokens 16384` for thinking variants is our choice (8192 risks truncating the trace).
  Truncation (`finish_reason != stop`, normally `length`): upstream `CosmosReason*` classes log it
  and score the raw, unparsed text. `CosmosReason3Think` instead parses it too: text after the last
  `</think>` if present, otherwise an **empty prediction**; both cases log a
  `[CosmosReason3Think] TRUNCATED` warning (count them in `run.log`) and are counted as
  `truncated` in `$INFERENCE_STATS_FILE` if set. The empty prediction is returned without
  triggering BaseAPI's retry loop (one request per sample).
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
- Caveat: `CosmosReason._get_video_processing_kwargs` drops **all** model-level sampling values
  as soon as a video message carries **any** sampling key (even only `max_pixels`). Checked
  (`scripts/check_video_sampling.py`, AST of each class's `build_prompt`): VsiBench, STIBench,
  SiteBenchVideo and DSRBench build `{type: 'video', value: path}` with no sampling keys, so our
  values apply. **MMSIVideoBench never sends a video message**: it subsamples the provided
  per-clip frame lists with the *dataset's* `nframe` (one global linspace over all segments)
  and sends the frames as images, so its frame count is the dataset-side `nframe` and it has no
  `video_sampling` block. The same script also proves the counts dynamically (synthetic
  40 s / 30 fps video -> client preprocessing -> frames in the request): VSI-Bench(-Debiased) 32,
  STI-Bench 30, SiteBenchVideo 32, DSRBench 40 (1 fps), MMSIVideoBench 64 of 300.
  `run_suite.py` refuses configs that set both `nframes` and `fps` (or dataset `nframe` and `fps`).

## 6. Per-benchmark settings

Frames: `uniform` = `np.linspace(0, n-1, k)` (VSI-Bench paper / lmms-eval convention; the
client's qwen-vl-utils `nframes` sampling is also uniform linspace).

| bench | suite | modality | frames | scoring | judge | frame source |
|---|---|---|---|---|---|---|
| VSI-Bench | spatial, smoke | video | 32 uniform | MCQ rule + NA (MRA) | exact_matching | task spec; VSI-Bench paper; kit variant `VSI-Bench_32frame` |
| VSI-Bench-Debiased | spatial | video | 32 uniform | as VSI-Bench | exact_matching | same as VSI-Bench |
| MMSIVideoBench | spatial | video (sent as image frames) | 64 (dataset-side) | MCQ rule | exact_matching | kit variant `_64frame`; paper SC policy = 300 frames (TODO) |
| STI-Bench | spatial | video | 30 | MCQ rule | exact_matching | official paper setting (kit comment) |
| SiteBenchVideo | spatial | video | 32 | MCQ rule | exact_matching | kit variant `_32frame` (no official value annotated) |
| DSRBench | spatial | video | 1 fps | MCQ rule | exact_matching | official paper setting (kit comment) |
| ReVSI_64frame (also _32frame, _16frame, _tiny) | spatial | video | 64 official pre-sampled | NA MRA + MCQ exact match (official parser) | none (ignored) | official `video.zip:64_frame/*.mp4`; §6.1 |
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

### 6.1 ReVSI

Source: `vlmeval/dataset/revsi.py`, a port of the official evaluation code (upstream VLMEvalKit
`revsi.py` @ `f5296f0`, lmms-eval `tasks/revsi` @ `9134731`, torchmetrics_ext `ReVSIMetric` @ `3c18d78`;
all three agree). Data: HF `3dlg-hcvc/ReVSI` @ `345a811` in `$LMUData/ReVSI` (parquet subsets,
`metadata/`, `video.zip`; only the needed `<N>_frame/` folder is extracted to `$LMUData/ReVSI/videos/`).

| key | parquet subset | questions | tasks | paper Qwen3-VL-8B / 32B Avg |
|---|---|---|---|---|
| `ReVSI_64frame` (spatial suite) | `64_frame` | 6,616 | 7 | 49.1 / 53.3 (main table) |
| `ReVSI_32frame` | `32_frame` | 6,158 | 7 | 45.6 / 51.2 (Table 12) |
| `ReVSI_16frame` | `16_frame` | 4,568 | 5 (no Room Size, Route Plan) | 43.7 / 46.2 (Table 13) |
| `ReVSI_tiny` | `64_frame` ∩ `metadata/tiny_set_question_ids.txt` | 1,057 (of 1,093 ids) | 7 | – (paper: proprietary models only) |

Per-task counts, 64-frame: Obj. Count 1,090 (single 694 / multiple 396), Abs. Dist 1,208, Obj. Size 1,126,
Room Size 302 (174 / 128), Rel. Dist 1,279 (closest 760 / farthest 519), Rel. Dir 1,332
(fwd-easy 400, bwd-easy 258, fwd-hard 354, bwd-hard 320), Route Plan 279. The question set differs per
frame budget (questions about objects not visible in the N frames were removed by the authors).

- **Frames.** The official N-frame videos are pre-sampled by the authors (hierarchical uniform
  sampling of the all-frame video: 64 ⊂ all, 32 ⊂ 64, 16 ⊂ 32, indices in
  `metadata/sampled_video_frame_idx.json`) and shipped as `video.zip:<N>_frame/<scene>.mp4` with
  exactly N frames. `ReVSI.build_prompt` sends that mp4 as a video message **without sampling
  keys**; the data config sets `video_sampling.nframes = N`, and qwen-vl-utils'
  `linspace(0, N-1, N)` over an N-frame video is the identity, so every official frame is sent once, in
  order (fps sent = the mp4's fps, i.e. the frames' original timestamps). Proof on the real files:
  `scripts/check_revsi_frames.py` (see below); synthetic: `scripts/check_video_sampling.py`.
  Evidence (2026-11-25, `cosmos3_nano_nothink` layer, endpoint stubbed): for sampled scenes of every key the
  request carries exactly N frames (64/32/16) at native size (640x480 / 480x640), request frame i ==
  mp4 frame i for all i, `mm_processor_kwargs={do_sample_frames: false, fps: <mp4 fps>}`; the image path
  yields N images. Provenance vs `sampled_video_frame_idx.json` (all-frame video): exact for every
  64-frame scene checked (7); some 32/16-frame encodes of long ARKitScenes videos match a neighbouring
  all-frame frame (offsets -1..-5, <= 0.5 s) - a property of the official files, which upstream sends too.
  `video.zip` sha256 `7d10f8a5...eabb9fe` (= HF LFS oid).
  Non-VIDEO_LLM clients get all N frames as images. The class rejects `nframe != N` and `fps > 0`.
- **Prompt** (official, verbatim): `These are frames of a video.\n<question>\n` + numeric:
  `Answer the question using a single integer or decimal number.`; MCQ:
  `Options:\n<A. ...>\n...\nAnswer with the option's letter from the given choices directly.`
  Video before text. `TYPE='MCQ'` (upstream: `Video-VQA`) so that `CosmosReason*.use_custom_prompt`
  returns False and this prompt is used for every question.
- **Metric** (official): numeric -> MRA, `mean_c 1[|p-t|/t <= 1-c]`, c in 0.50:0.05:0.95 (10 thresholds);
  MCQ -> case-insensitive exact match. Task = mean of its sub-type means (Obj. Count = single, multiple;
  Room Size = single, multiple; Rel. Dist = closest, farthest; Rel. Dir = 4 sub-types). **Avg = mean of
  the task columns present** (7; 5 at 16 frames). Reported 0-100.
- **Answer parsing.** `*_official` keys (`Overall_official`, `<task>_official`) = the official parser
  bit-for-bit (first whitespace token, `rstrip('.')`; letter match / `float()` + MRA; anything else 0).
  Headline `Overall` / `<task>` = same, but when the official token is not a valid answer (not one of
  the option letters / not a finite number) it falls back to the kit's rule extractors
  (`can_match_option`, `can_match_na` on the text after the last `</think>`; thousands separators
  removed), e.g. `(B)`, `The answer is B.`, `3m`, `about 2.5 meters`, `four`. Where the official parse
  succeeds both are identical, so for format-following models `Overall == Overall_official`; compare
  `Overall_official` with the paper when they differ. No unit conversion (the questions state the unit).
  Per-row results: `<eval>_revsi_score.xlsx` (`parse_mode` official/fallback/fail); per-task table
  `<eval>_acc.csv` (n, score, score_official, n_fallback, n_unparsed).
- **No judge.** `evaluate()` ignores `--judge`.
- **Differences from the paper setup** (besides the client/server of §1-3): the paper ran Qwen3-VL
  through SWIFT / lmms-eval (HF transformers or vLLM, greedy, lmms-eval `max_new_tokens 16`); here the
  model defaults apply (greedy, `max_tokens` 8192/16384) and frames are re-encoded by the client at
  qwen-vl-utils default resolution limits (ReVSI frames are 640x480 / 480x640 / 360x640, below the
  786,432 px per-frame cap, so native resolution). `all_frame` (2 fps / variable length) is not provided.
- **Tests:** `scripts/test_revsi_scoring.py` (offline: GT -> 100 for all subsets, garbage -> 0, random
  -> chance, 20 hand-made parse cases, MRA vs upstream formula).

## 7. Reproducibility caveats

- QSpatial downloads its prompt templates from the Q-Spatial-Bench GitHub `main` branch at run time (not pinned).
- Dataset TSVs are verified by the MD5 in each dataset class where one is given; HF-hosted
  datasets (Omni3DBench, CoreCognition, PhyX, SeePhys, Q-Spatial) are fetched from `main`.
- Smoke runs (`sample_limit`) score a seeded random subset (seed 0); never report them.
