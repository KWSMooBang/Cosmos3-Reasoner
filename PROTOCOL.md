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

Deviations found / applied in the t003 smoke run (vLLM 0.32.0+cu129, Cosmos3-Nano, TP=1, 1× RTX 4090):

| deviation | why | evidence |
|---|---|---|
| `--trust-request-mm-kwargs` added | vLLM 0.32 returns HTTP 400 for any request with `mm_processor_kwargs` without it; the clients send `{fps, do_sample_frames: false}` with every video (VSI-Bench 20/20 failed, score 0) | t003 run 1 |
| `--media-io-kwargs` video adds `"video_backend": "opencv"` | vLLM 0.32's `qwen3_vl` loader turns `fps=-1` into `min_frames=4`; a 32-frame client video reached the model as 4 frames (2 temporal groups) | t003 run 2; offline loader check |
| TP=1, `--max-model-len 16384`, image/video `max_pixels` caps | 24 GB GPU (serving/README memory budget). KV cache measured: 20,736 tokens; GPU 1 used 21,201 MiB | t003 |
| think configs: requested `max_tokens=16384` is clamped by the server to `16384 − prompt_len` (≈11.3k for 32-frame VSI, ≈12.5k–16k for MindCube) | max-model-len 16384 | vLLM request log |

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
| VSTIBench | spatial | video | 32 uniform (pinned in message) | NA MRA + MCQ exact match (official parser); Overall = paper Avg over 5 tasks | none (ignored) | official `eval_vlm_3r_vstibench.sh`; see "VSTI-Bench" below |
| 3DSRBench_official | spatial | image | – | MCQ rule extraction; Overall = official CircularEval (flip copies separate) | exact_matching | see "3DSRBench" below |
| MMSIBench_wo_circular, OmniSpatialBench, ViewSpatialBench, MindCubeBench_raw_qa (smoke: _tiny), SiteBenchImage, EmbSpatialBench, ERQA, BLINK, MMVP | spatial | image | – | MCQ rule extraction | exact_matching | – |
| SparBench | spatial | image | – | MCQ rule + NA | exact_matching (rule path) | – |
| Spatial457, Omni3DBench | spatial | image | – | dataset-internal rule metric | none used | – |
| RefSpatial, RoboSpatialHome | spatial | image | – | Point2D in-mask | none used | – |
| QSpatial_plus / _scannet | spatial | image | – | rule extraction (no judge passed) | none (a judge would switch it to LLM extraction) | – |
| CoreCognition | spatial | image | – | hybrid matcher, exact-match fallback | exact_matching (forces fallback; NVIDIA run.py default would be gpt-4.1) | – |
| SpaceCastBench_CoT (default), SpaceCastBench (direct) | spatial (direct opt-in) | 1–8 images | – | official SpaceCast parser + exact set match, macro over 16 types | none | see "SpaceCast-Bench" below |
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
  exactly N frames. `ReVSI.build_prompt` sends that mp4 as a video message that **pins
  `nframes=N`**, so the budget holds for any model config and for direct `run.py` use. By the
  §5 rule, a sampling key in the message makes `CosmosReason*` ignore **all** model-level sampling
  attributes (nframes/fps/max_pixels/total_pixels/min_pixels/max_frames) for ReVSI. Resolution
  therefore uses the qwen-vl-utils defaults, which keep ReVSI's 640x480 frames at native size; a
  model-level `max_pixels` etc. would not apply here (`check_video_sampling.py` fails if a model
  layer sets one for ReVSI). The data config still carries `video_sampling.nframes = N`, which
  must equal the pinned key. qwen-vl-utils' `linspace(0, N-1, N)` over an N-frame video is the
  identity, so every official frame is sent once, in order (fps sent = the mp4's fps, i.e. the
  frames' original timestamps). Proof on the real files: `scripts/check_revsi_frames.py` (also with no
  `video_sampling` and with conflicting model-level nframes=8 / fps=2 / max_frames=4: still N frames
  in order); synthetic: `scripts/check_video_sampling.py`.
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
- **Answer parsing (headline = official parser; L1 decision 2026-11-25).** Headline `Overall` /
  `<task>` = the official parser bit-for-bit (first whitespace token, `rstrip('.')`; letter match /
  `float()` + MRA; anything else 0), so it is directly comparable with the paper. This is what
  `parse_score.py` reports as `overall`, and what `run_suite.py` and `results/summary.csv` show.
  Diagnostic `Overall_lenient` / `<task>_lenient` = same, but when the official token is not a valid
  answer (not one of the option letters / not a finite number) it falls back to the kit's rule
  extractors (`can_match_option`, `can_match_na` on the text after the last `</think>`; thousands
  separators removed), e.g. `(B)`, `The answer is B.`, `3m`, `about 2.5 meters`, `four`. Where the
  official parse succeeds both are identical, so `Overall_lenient - Overall` = points lost to answer
  format (expect ~0 for format-following non-thinking models). `run_suite.py` stores it as
  `score_lenient` in `score.json`; `results/summary.csv` has it in `overall_lenient` (empty for
  other benches). No unit conversion (the questions state the unit). Per-row results:
  `<eval>_revsi_score.xlsx` (`score`, `score_lenient`, `parse_mode` official/fallback/fail); per-task
  table `<eval>_acc.csv` (n, score, score_lenient, n_fallback, n_unparsed).
- **No judge.** `evaluate()` ignores `--judge`.
- **Upstream quirks (kept for parity).** MRA is computed exactly as upstream:
  `abs(p - t) / t <= 1 - np.linspace(0.5, 0.95, 10)` in float64. A relative error that sits exactly on a
  threshold passes or fails depending on float rounding: `120` vs `100` scores 0.6 (0.2 > 1-0.80 =
  0.19999999999999996), while `1.2` vs `1.0` scores 0.7 and exact maths would give 0.7 for both. We do not
  fix this, so our numbers match the official scorer bit-for-bit (tested in `test_revsi_scoring.py`).
- **Differences from the paper setup** (besides the client/server of §1-3): the paper ran Qwen3-VL
  through SWIFT / lmms-eval (HF transformers or vLLM, greedy, lmms-eval `max_new_tokens 16`); here the
  model defaults apply (greedy, `max_tokens` 8192/16384) and frames are re-encoded by the client at
  qwen-vl-utils default resolution limits (ReVSI frames are 640x480 / 480x640 / 360x640, below the
  786,432 px per-frame cap, so native resolution). `all_frame` (2 fps / variable length) is not provided.
- **Data integrity:** the extraction marker is trusted only if every mp4 the subset references exists
  and is non-empty; otherwise `video.zip` is checked against its sha256 (deleted and re-downloaded
  on mismatch) and the subset is re-extracted. Frame JPEGs (image path) are written atomically.
- **Tests:** `scripts/test_revsi_scoring.py` (offline: GT -> 100 for all subsets, garbage -> 0, random
  -> chance, 20 hand-made parse cases, MRA vs upstream formula).

### SpaceCast-Bench

Local class `vlmeval/dataset/spacecast.py`; the official helpers are a verbatim copy in
`vlmeval/dataset/utils/spacecast_official.py`. Sources: data HF `hongxingli/SpaceCast-Bench` @
`45be662c` (parquet sha256 `27541adf…`), official evaluator `ZJU-REAL/SpaceCast-Bench` @ `6696e5eb`
(protocol `spacecast-v1`), arXiv 2610.12402.

- Data: 3,862 MCQ from 182 ScanNet / ScanNet++ scenes, 16 task types (L1 static 6 types / 1,368 questions,
  L2 local 7 / 1,709, L3 global 3 / 785). Per-category counts are equal to the official
  `RELEASE_CATEGORY_COUNTS` (checked when the TSV is built). Categories are `"<level> / <type>"`,
  so L2 and L3 "Rotate Object Direction" stay separate. On first load the parquet is converted to
  `$LMUData/SpaceCastBench.tsv` (both keys share it) plus `$LMUData/images/SpaceCastBench/<dataset>/<scene>/<frame>`.
- Cache integrity: the build writes `$LMUData/SpaceCastBench.manifest.json` (HF repo/revision, parquet
  sha256, parquet path used, build version, TSV md5, sha256 of every referenced image). Every load
  re-validates the cache against the manifest (≈0.4 s): pinned revision, TSV md5, image list equal to the
  images the TSV references, every image present with the same sha256. On any mismatch the cache is
  rebuilt. Building and validation run under `fcntl.flock` on `$LMUData/SpaceCastBench.lock`, and every
  temp file is unique per process (pid + uuid), so direct and CoT runs can cold-start concurrently
  (tested: one process builds, the other waits and then validates). The parquet is the local copy
  `$LMUData/SpaceCast-Bench/hf/test-00000-of-00001.parquet` if present, else `hf_hub_download` at the
  pinned revision. Either way its sha256 must equal the pinned LFS oid `27541adf…`, otherwise the build stops.
- Keys: **`SpaceCastBench_CoT` = official `--mode cot` (default; the official main experiment)**: brief
  reasoning, last line `Answer: <letter>`. `SpaceCastBench` = official `--mode direct` (suffix "Answer with a
  single letter only (A, B, C, or D). Do not explain."). The spatial suite runs CoT; direct is a commented opt-in line.
- Request: system prompt "You are a careful vision-language assistant solving multiple-choice spatial
  reasoning questions about an image.", then one user turn `[prompt text, image_1 … image_n]` (text
  first, images after it with no labels, as in the official OpenAI-compatible backend). Prompt = question, blank line,
  `A) …` option lines, blank line, mode suffix (multi-select suffix for Attachment). Image order =
  main view A, bridge views, main view B.
- Images: the official `prepare_image` (RGB, longest side 1280, JPEG quality 90) is applied once
  when the TSV is built, and the JPEG bytes are sent unchanged (`img_size=-1`). This is the one image
  bench that does not send native resolution. View sizes after this: 1280×960 / 1280×956 / 1024×768 / 640×480.
  With the Cosmos3 / Qwen3-VL processor (patch 16, merge 2 → 32×32 px per token) a 1280×960 view
  is 1,200 tokens. Images per question: 1 (534 q), 3 (1,092), 4 (1,118), 5 (726), 6 (275), 7 (86), 8 (29); mean 3.7.
  Image tokens per question: mean ≈3.5k, max 9,600. The full prompt of the largest question,
  measured with the Cosmos3-Nano HF processor, is 9,856 (direct) / 9,892 (cot) tokens. That fits
  `max-model-len 16384` at TP=1, so **no `max_pixels` is set**. The TP=1 server cap of 8,388,608 px per image does not
  apply (1280×960 = 1.2 MP), and vLLM's default per-prompt image limit is 999. vLLM 0.32 clamps
  `max_tokens` to the remaining context, so at TP=1 an 8-view question leaves ≈6.5k output tokens. That is enough for
  direct and short cot replies, but `*_think` runs (16384 max_tokens) can truncate on 7–8-view questions.
  For thinking runs use TP=2 / 65536. A `max_pixels` of 614,400 (≈600 tokens/view) would be the
  fallback if TP=1 had to be used, but it departs from the official 1280 px.
- Scoring: official `parse_response` (letter, `Answer: X`, trailing letter, option text; multi-select letter
  lists) → `parse_prediction` → exact set match. Unparsable or API-failed answers count as wrong.
  Headline `Overall` = macro average over the 16 categories (official `macro_type_accuracy`). Also reported:
  `micro_accuracy`, `L{1,2,3}_macro` (mean of that level's type accuracies), `L{1,2,3}_micro`, per-category
  accuracy, `n_invalid`, `n_error`, `n_lenient_parse` (see quirks below) and `evaluation_scope`.
  `evaluation_scope` is `full` only when the scored set is the release: 3,862 rows, unique question ids, the
  sorted id set hashing to the pinned release fingerprint, the per-category counts equal to
  `RELEASE_CATEGORY_COUNTS`, and 182 scenes. Anything else (e.g. `--sample-limit` runs) is `subset`. The official
  scorer withholds the macro for partial runs; here it is computed over the categories present. No LLM judge.
- Parity (`scripts/check_spacecast_parity.py`, no GPU): over all 3,862 questions, prompts (both modes) and
  image order equal the official ones, sampled image bytes (760 images) equal the official `prepare_image`,
  and official `scoring.score` and ours agree per question on a 20-template synthetic response set
  (0 mismatches, macro 56.72 both; 36 of them flagged `lenient_parse`: `Answer: <option text>`
  where the text starts with a valid letter). GT-as-prediction gives 100. It also checks that the parquet the builder
  recorded in the manifest has the pinned sha256, that `release_check` passes on the cache and rejects
  subset / duplicate-id / foreign-id sets, and the `lenient_parse` flags on the two quirk examples.
- Differences from the official run (`spacecast_eval/run.py`):
  1. The system prompt reaches the model through `BaseAPI.preprocess_message_with_role`, which appends
     `"\n"` to it. With `CosmosReason3Think` (`*_think`) it is appended to that class's own
     "You are a helpful assistant." system prompt, and the Cosmos3 think instruction is appended to the user
     text. The official runs have exactly one system prompt.
  2. direct mode: the official client re-asks up to 2 times when a reply has no parseable letter, then records
     an error. We do not re-ask (with greedy decoding the reply would be the same); the question scores wrong in both cases.
  3. cot mode: the official client puts a separately returned `reasoning_content` in front of the answer.
     The kit uses `content` only (and `*_think` drops text before `</think>`). Only the final
     `Answer:` line is parsed, so this matters only when the answer appears in the reasoning alone.
  4. `max_tokens` / temperature come from our model layer (greedy T=0 like the paper; official
     default `max_tokens` 3072, ours 8192 nothink / 16384 think, clamped by vLLM as above).
  5. Files are written under their official `image_paths` (all 2,702 unique views are `.jpg`; a
     non-`.jpg` name would get `.jpg` appended). Built TSV md5 `53e6a9a00a949cde2f49f49790b220b7`, images 365 MB.
- Upstream parser quirks (kept for parity, scoring unchanged, counted as a diagnostic): the official
  `parse_answer` / `parse_answers` accept some answers leniently.
  1. Out-of-range letters are dropped: `Answer: A, Z` → {A} (multi- and single-select).
  2. The letter-capture regex has no word boundary, so the first letter of a word after `Answer:`
     becomes the answer: `Answer: Blue chair` → B, even when "Blue chair" is the text of option C
     (the option-text fallback is only reached when no letter pattern matches).
  Each scored question gets a `lenient_parse` flag in `*_score.xlsx` (`out_of_range`, `letter_from_word`, or
  `letter_from_word_conflict` when the segment equals another option's text). The score JSON reports
  `n_lenient_parse` and `n_lenient_parse_correct` (how many of those are counted correct). A large
  count means the headline depends on these quirks; report it next to the score.

### VSTI-Bench

Local class `vlmeval/dataset/vstibench.py` (key `VSTIBench`), a port of the official evaluation code
VITA-Group/VLM-3R `thinking-in-space/lmms_eval/tasks/vstibench/{vstibench.yaml,utils.py}` @ `206afd0e`
(prompt, question types, parser, MRA, aggregation verbatim). Paper: VLM-3R, arXiv 2505.20279 v5 (CVPR 2026), Table 2.
Data: HF `Journey9ni/vstibench` @ `0903eb0b` (linked from the official repo and project page), in
`$LMUData/VSTIBench/` (`test.json` sha256 `7c0780c0…`, `ScanNet.tar.gz` 5.53 GB sha256 `33339aa0…` = LFS oids;
extracted mp4s 5.2 GB). No licence on the card; the videos are ScanNet val scans (ScanNet Terms of Use).

- **Data.** 5,736 questions on 312 ScanNet val videos (all 640×480, 24 fps, 295–5,422 frames, mean 1,724).
  This is the **post-erratum** release (2026-07-14: camera-pose bug fixed, 306 questions dropped; the
  pre-erratum release @ `27ff0a2` had 6,042). The paper's Table 2 was computed on the pre-erratum labels, and the
  repo README says those numbers are not comparable (the authors plan to re-report). Counts @ `0903eb0b` (checked
  when the TSV is built):

  | task (paper column) | question types (n) | answer | metric |
  |---|---|---|---|
  | Cam-Obj Abs. Dist | `camera_obj_abs_dist` (905) | number (m) | MRA |
  | Cam. Displace | `camera_displacement` (833) | number (m) | MRA |
  | Cam. Mov. Dir | `camera_movement_direction` (913; 2/3/4 options) | letter | Acc |
  | Obj-Obj Rel. Pos | `obj_obj_relative_pos_{lr,nf,ud}` (605/556/484; 2 options) | letter | Acc |
  | Cam-Obj Rel. Dist | `camera_obj_rel_dist_v{1,2,3}` (91/493/856; 4/3/2 options) | letter | Acc |

- **Frames.** 32 uniform frames (official `eval_vlm_3r_vstibench.sh`: `max_frames_num=32`; `vlm_3r.load_video` =
  `np.linspace(0, n-1, 32, dtype=int)` over the mp4). The paper states no frame count for the baselines. The
  questions refer to "frame k of 32" (5,292) or "of N<32" (444), counted over the authors' 32 valid-pose `.sens`
  frames, so the frames match those indices only approximately, for the authors' own model as well. `build_prompt`
  sends the mp4 as a video message that **pins `nframes=32`** (as ReVSI does, so model-level sampling attributes are ignored and
  qwen-vl-utils' default limits apply). qwen-vl-utils samples `torch.linspace(0, n-1, 32).round()`, which differs from the
  official floor by at most one source frame (1/24 s). Evidence (`scripts/check_vstibench_frames.py`, 2026-11-26, 6 videos ×
  5 model configs including conflicting model-level nframes=8 / fps=2 / max_frames=4): every request carries 32
  frames at native 640×480. Sent frame vs official index: offset 0 for 585 frames and +1 for 375, never anything else.
  `mm_processor_kwargs={do_sample_frames: false, fps: 32/n·24}`. The non-VIDEO_LLM path sends exactly the official
  indices as images.
- **Tokens (Cosmos3-Nano processor @ `45666784`).** 640×480 is 20×15 tokens per 2-frame group, i.e. **4,800 video
  tokens for every question** (all videos have the same size). The full prompt of the longest question of each type is
  4,997–5,040 tokens. That fits `max-model-len 16384` at TP=1 and leaves ≈11.3k for the reply, the same as 32-frame
  VSI-Bench (`*_think` 16384 max_tokens is clamped by vLLM).
- **Prompt** (official, verbatim; yaml `pre_prompt ""` falls back to the utils default):
  numeric `These are frames of a video.\n<question>\nPlease answer the question using a single word or phrase.`;
  MCQ `These are frames of a video.\n<question>\nOptions:\nA. …\n…\nAnswer with the option's letter from the given choices directly.`
  Video before text. `TYPE='MCQ'` so that `CosmosReason*.use_custom_prompt` returns False and this prompt is used.
- **Metric** (official): parser `pred.split(' ')[0].rstrip('.').strip()`. MCQ: case-insensitive equality with `mc_answer`.
  Numeric: `float()`, then MRA `mean_θ 1[|p−t|/t ≤ 1−θ]` with θ = 0.50:0.05:0.95 (10 thresholds). Unparsable answers score 0. The same
  float-threshold quirk as ReVSI is kept for parity (e.g. 2.4 vs 2.0 → 0.7).
  - **Headline `Overall` = the paper's Avg**: the mean of the 5 task columns, where Obj-Obj Rel. Pos and Cam-Obj Rel. Dist are each
    the mean of their sub-type means. The paper does not state this; the evidence is that its chance row has Cam-Obj Rel. Dist = 36.1 =
    mean(1/4, 1/3, 1/2) (the pooled value would be 42.8), and that VLM-3R / Gemini Avg = the mean of the 5 columns.
  - `Overall_code9` = what the official lmms-eval aggregator returns. It is the unweighted mean of the 9 per-question-type scores and does not
    merge sub-types, so it differs from the paper's Avg.
  - Also reported: the 5 task scores and 9 per-type scores. As for ReVSI, `Overall_lenient` is a diagnostic: the kit's rule extractors
    are used only when the official token is not a valid answer.
  - No judge (`evaluate()` ignores `--judge`). Per-row results go to `<eval>_vstibench_score.xlsx` and the task table to `<eval>_acc.csv`.
  - **Coverage.** `evaluation_scope` is `full` only when the scored rows are exactly the release: 5,736 unique ids equal
    to the TSV's, all 9 types at the release counts. Anything else (e.g. `--sample-limit`) is `subset`, with
    `headline_partial=1` and a warning. `run_suite` then marks the row "PARTIAL headline". Overall is still the mean
    over the tasks present, so it is not the release metric. Also reported: `n_scored`, `n_types`, `n_duplicate_rows`.
  - **Zero targets.** The official MRA divides by the target, and a target of 0 raises an uncaught `ZeroDivisionError`
    upstream. The release has **0** zero targets (numeric targets 0.2–8.2 m), so our port keeps that behaviour,
    and `n_zero_target` is reported (always 0 on the release).
- **Cache integrity.** The first load writes `$LMUData/VSTIBench.manifest.json`: HF repo/revision, build version, pinned
  sha256 of `test.json` (`7c0780c0…`, whose git blob id equals the Hub's `bc83433b`) and `ScanNet.tar.gz` (`33339aa0…`
  = LFS oid), the TSV md5 (`d7c24a5f2d1c66bb3283114eee67718d`), and size + sha256 of each of the 312 mp4s.
  - Every load re-validates against the manifest (≈5 s): pins, `test.json` sha256, TSV md5, the video list equal to the
    one `test.json` references, and every video's size and sha256.
  - On any mismatch the cache is rebuilt under a lock. A file with the wrong hash is deleted and re-downloaded at the
    pinned revision; the mp4s are re-extracted and the TSV rebuilt. If the cache is still invalid after that, the load fails.
  - `test_vstibench_scoring.py` checks that tampered revision, build version, TSV, video hash, video size and video
    list are all rejected.
- **Parity / tests** (`scripts/test_vstibench_scoring.py`, offline): GT → 100 (Overall, code9, lenient); garbage → 0.
  The official `utils.py`, imported unchanged, agrees with ours on a 33-template synthetic response set over all 5,736
  questions: 0 per-question mismatches, and its aggregate equals `Overall_code9` (29.750713 both). The paper Avg matches an
  independent recomputation. 21 hand-made parse cases pass.
- **Differences from the official run:** the official generation is `max_new_tokens 16`, greedy; ours uses the model
  layer (greedy, 8192/16384). Our client re-encodes the 32 frames (round vs floor indices, see above) and sends
  timestamps via `fps`; the official LLaVA-based loader sends bare frames.
- **Paper reference (Table 2, pre-erratum labels, not directly comparable).** Columns: Avg / Cam-Obj Abs / Cam Disp / Cam Mov Dir /
  Obj-Obj Rel Pos / Cam-Obj Rel Dist.
  - VLM-3R (7B): 58.8 / 39.4 / 39.6 / 60.6 / 86.5 / 68.6
  - GPT-4o: 38.2 / 29.5 / 23.4 / 37.3 / 58.1 / 42.5
  - Gemini-2.5 Pro: 42.4 / 29.2 / 8.0 / 30.0 / 84.0 / 60.7
  - LLaVA-NeXT-Video-7B 40.0, LLaVA-OneVision-7B 41.7, InternVL2-8B 43.5, LLaVA-NeXT-Video-72B 44.0
  - Chance (frequency) 27.4, human 77.0
  - **No Qwen2.5-VL or Qwen3-VL numbers** are reported for VSTI-Bench (Qwen2.5-VL appears only in the VSI-Bench table).

### 3DSRBench

Paper arXiv 2412.07825 v4; data HF `ccvl/3DSRBench` @ `5c90e422` (CC-BY-4.0). Only the real split (MS-COCO train2017 images, original
resolution, longest side 640) is released, not the HSSD synthetic split.

- **TSV.** The OpenCompass `3DSRBench.tsv` (md5 `610516a0b4710595545b7613c60524e8`, 262 MB, `$LMUData/3DSRBench.tsv`) has
  the same rows as the authors' `3dsrbench_v1_vlmevalkit_circular.tsv`, with images de-duplicated.
  - 11,686 rows = 2,625 base questions (2,100 annotations, 350 each for height / location-above / location-closer after the
    higher↔lower augmentation) + 2,532 horizontally flipped copies (`<id>-flip`, left/right answers swapped).
  - That gives 5,157 questions, each repeated once per option ordering (`""`, `-1`, `-2`, `-3`): 2 orderings for the 2-option
    questions, 4 for the 4-option `*_viewpoint*` ones.
  - 12 sub-types in 4 categories: height 1, location 3, orientation 3, multi-object 5.
  - Download: the `opencompass.openxlab.space` TLS certificate has expired (2026-11). The file was fetched from the
    same Aliyun OSS bucket (`opencompass.oss-cn-shanghai.aliyuncs.com/utils/VLMEval/3DSRBench.tsv`, valid certificate) and
    verified by md5. `3DSRBench_official` tries both URLs and requires that md5.
- **Official metric (paper + authors' `compute_3drbench_results_circular.py`).** CircularEval: a question is
  correct iff every option ordering is answered correctly (the scorer strips a trailing `-N` from `qid` and multiplies the hits).
  `-flip` copies stay **separate questions** (FlipEval acts as augmentation, not as pairing).
  Overall and the 4 categories are means over questions. The paper's random row confirms this: height 25.0 = (1/2)²; with flip pairing it would be about 6.25.
  The paper extracts answers MMBench-style: exact matching, falling back to GPT.
- **Verdict: the previous setup differed. Fixed.** The upstream key `3DSRBench` (`_3DSRBench.evaluate`) scores all
  rows and returns a 4-row DataFrame (`vanilla`, `flip_eval`, `circ_eval`, `flip_circ_eval`). Its `circ_eval` row equals
  the official metric. However, run.py transposes it into a `.df.eval.json`, and `parse_score` (= `run_suite`'s headline)
  reads that as **0.00 for every prediction set**, including GT (`scripts/test_3dsrbench_official.py`). Even if
  parsing were fixed, the generic first-row rule would pick `vanilla`, not the paper metric.
  - New key **`3DSRBench_official`** (`vlmeval/dataset/threedsrbench_official.py`) uses the same TSV, prompt and extraction
    as upstream. It returns a flat dict: `Overall` = official CircularEval, plus height / location / orientation /
    multi_object, the 12 sub-types, `n_questions`, `evaluation_scope` (`full` iff 11,686 rows / 5,157 questions), and the
    diagnostics `vanilla`, `flip_eval`, `flip_circ_eval`, `upstream_circ_eval`. The spatial suite now runs this key.
    The upstream class is untouched, and the key `3DSRBench` stays available.
  - **Copy validation.** The official script multiplies whatever copies are present. A question with one correct copy and
    its other ordering(s) missing would therefore count as correct, and a duplicated row would be multiplied in.
    `3DSRBench_official` first checks the scored rows against the TSV's copy structure (expected copies = every TSV qid
    with the same group key).
    - Questions with a missing ordering or a duplicated copy are **excluded** from Overall and the categories, and flagged:
      `n_incomplete_excluded`, `n_duplicated_excluded`, `n_duplicate_rows`. Qids that are not in the TSV raise an error.
    - `evaluation_scope` is `full` (`headline_partial=0`) only if all 5,157 questions are scored with every copy exactly
      once. Otherwise it is `subset` with `headline_partial=1`, a warning, and `run_suite` marks the row "PARTIAL headline".
      `n_missing_questions` is reported too.
    - `Overall_present_copies` = the official script on the rows as given (diagnostic). On the full release both are identical.
    - Upstream `_3DSRBench.evaluate` raises `KeyError` on such partial inputs (e.g. `<id>-flip-1` present but `<id>-1`
      missing). `3DSRBench_official` therefore reproduces only its hit computation (same judge handling,
      `mcq_vanilla_eval`, `_result.xlsx`) and recomputes the four upstream settings itself. On full inputs,
      `upstream_circ_eval` equals the upstream key's `circ_eval` row (tested).
- **Prompt / images.** The authors' README runs `run.py --data 3DSRBenchv1`, i.e. VLMEvalKit's default
  `ImageMCQDataset.build_prompt` (`Question: …\nOptions:\nA. …\nPlease select the correct answer from the options above. \n`).
  `CosmosReason*` uses exactly this for MCQ datasets (`use_custom_prompt` → False). Images are sent at native resolution (`img_size=-1`).
- **Parity** (`scripts/test_3dsrbench_official.py`, offline, ~3 min). The authors' scorer runs verbatim on our `_result.xlsx`,
  laid out like its `_openai_result.xlsx`. Over 8 prediction sets (GT, garbage, all-A, random ×2, GT with 30 % of rotated / flipped
  copies wrong, mixed text answers), ours matches overall, the 4 categories and 12 sub-types (0 mismatches at the
  script's 0.1 rounding), and equals upstream `circ_eval` exactly. Examples: random 22.1 / 21.3 (paper random 20.9); all-A 0.0.
  All full sets report `evaluation_scope=full`. Partial cases:
  - one correct copy with its other ordering missing: excluded, Overall 0 (the official script on present copies gives > 0);
  - a duplicated copy: excluded;
  - whole questions missing: subset, no exclusions;
  - an unknown qid: error;
  - end to end, `evaluate()` on a GT file with the `-1` copy of 50 questions dropped: 50 excluded, n = 5,107, subset, Overall 100.
- **Difference from the paper:** `--judge exact_matching` (no GPT fallback), so answers the rule extractor cannot parse
  count as wrong. The paper falls back to GPT-4 extraction.
- **Paper reference (v4 Table, real split, CircularEval).** Columns: Overall / Height / Location / Orientation / Multi-object.
  - Qwen2.5-VL-7B: 48.4 / 44.1 / 62.7 / 40.6 / 40.5
  - InternVL2.5-8B: 50.9 / 45.9 / 68.1 / 38.7 / 43.3
  - GPT-4o: 44.2 / 53.2 / 59.6 / 21.6 / 39.0
  - Gemini-2.0-Flash: 49.8 / 49.7 / 68.9 / 32.2 / 41.5
  - QwenVLMax: 52.0 / 45.1 / 70.7 / 37.7 / 44.8
  - SpatialReasoner: 60.3
  - Human: 95.7
  - **No official Qwen3-VL number**: the Qwen3-VL tech report and model cards do not list 3DSRBench.

## 7. Reproducibility caveats

- QSpatial downloads its prompt templates from the Q-Spatial-Bench GitHub `main` branch at run time (not pinned).
- Dataset TSVs are verified by the MD5 in each dataset class where one is given; HF-hosted
  datasets (Omni3DBench, CoreCognition, PhyX, SeePhys, Q-Spatial) are fetched from `main`.
- Smoke runs (`sample_limit`) score a seeded random subset (seed 0); never report them.
