# serving/: vLLM servers for Cosmos3 and the Qwen3-VL baselines

This machine runs inside a container with no docker and no nvidia-container-toolkit. It has 8× RTX 4090 24 GB
(shared) and driver 550.120, so CUDA ≤ 12.4. We therefore serve with a **native vLLM venv**. The docker
recipe is only recorded as a reference (see the docker reference section below).

| file | purpose |
|---|---|
| `setup_vllm_env.sh` | builds `<main checkout>/.venv-vllm` (uv, Python 3.12, `vllm==0.32.0+cu129`, torch `2.13.0+cu129`, wheels only) and checks it |
| `requirements-vllm.lock` | the exact resolved package set (197 packages incl. pip/setuptools, `uv pip freeze`); `setup_vllm_env.sh --from-lock` reinstalls it |
| `serve_cosmos3_nano.sh` | `nvidia/Cosmos3-Nano`, default TP=1 (TP ∈ {1,2,4,8}) |
| `serve_cosmos3_super.sh` | `nvidia/Cosmos3-Super`, default TP=4 (TP ∈ {4,8}) |
| `serve_qwen3vl.sh` | `Qwen/Qwen3-VL-8B-Instruct` (default, TP=1) / `Qwen/Qwen3-VL-32B-Instruct` (TP=4) |
| `wait_healthy.sh` | waits until `GET /v1/models` lists the expected model (and the port belongs to our server) |
| `download_weights.sh` | `hf download` with only the files vLLM reads (needs approval for the big models) |
| `_common.sh`, `_portcheck.py` | shared flag parsing / launcher (sourced), `/proc`-based port-owner check (no `ss` in this container) |

## Quick start

```bash
serving/setup_vllm_env.sh --from-lock           # once; ~11 GB venv, wheels only, ~10 min
# weights must already be in $HF_HOME (/cache/huggingface); the serve scripts set HF_HUB_OFFLINE=1
serving/download_weights.sh --model nvidia/Cosmos3-Nano --revision 45666784803b3a7ad577e0def405f9afcc3bddb2 --dry-run

serving/serve_cosmos3_nano.sh --gpus 0 --port 8080 --revision 45666784803b3a7ad577e0def405f9afcc3bddb2 \
  > logs/nano.log 2>&1 &
serving/wait_healthy.sh --model nvidia/Cosmos3-Nano --port 8080 --pid $! --timeout 1800
```

### setup_vllm_env.sh

- Default mode resolves `vllm==$VLLM_VERSION+$VLLM_CUDA` fresh and rewrites `requirements-vllm.lock`.
  `--from-lock` installs exactly the lock with `uv pip sync`. `--check-only` only runs the checks.
- Installs use `--only-binary :all:`, so an sdist that would need compiling fails instead of building.
  `uv pip sync --dry-run` of the lock into an empty venv resolves all 197 packages as wheels. The lock keeps `setuptools`: torch imports it at runtime, and `uv pip sync` removes anything unlisted.
- The checks are CPU only and fail on any mismatch. They require:
  - Python == `--python` (3.12);
  - `vllm` == `X.Y.Z+cu129` exactly;
  - `torch` == vLLM's own `torch==` pin + `+cu129` (2.13.0+cu129);
  - `torch.version.cuda` == 12.9;
  - both `Cosmos3ForConditionalGeneration` and `Qwen3VLForConditionalGeneration` in the registry;
  - the `vllm serve` entrypoint modules import cleanly.
- `--force` deletes the venv first, but only if the path resolves under `/workspace/Cosmos3-Reasoner`, has a
  `pyvenv.cfg`, and is not a symlink. The symlink check runs on the raw path, before `realpath`, so a link never
  leads to deleting its target.

### Serve scripts

All serve scripts take the same options; flags override env vars. Run any of them with `--help`.
`--dry-run` validates the options and prints the exact `vllm serve` command.

| option | env | default |
|---|---|---|
| `--model` | `MODEL` | per script |
| `--revision` | `REVISION` | (main) |
| `--port` / `--host` | `PORT` / `HOST` | 8080 / 127.0.0.1 |
| `--tp` | `TP` | Nano 1, Super 4, Qwen3-VL-8B 1, -32B 4 |
| `--gpus` | `CUDA_VISIBLE_DEVICES` | unset → `0..TP-1` |
| `--max-model-len` | `MAX_MODEL_LEN` | see the memory budget table |
| `--gpu-mem-util` | `GPU_MEM_UTIL` | 0.9 |
| `--max-num-seqs` | `MAX_NUM_SEQS` | vLLM default |
| `--video-frames` / `--video-fps` | `VIDEO_NUM_FRAMES` / `VIDEO_FPS` | -1 / -1 (reference: all frames, native fps) |
| `--mm-processor-kwargs` | `MM_PROCESSOR_KWARGS` | TP=1 8B-class: pixel caps (below); otherwise none. `none` disables them |
| `--fp8` | `FP8=1` | off |
| `--allow-download` | `ALLOW_DOWNLOAD=1` | off: `HF_HUB_OFFLINE=1`. On: `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are unset |
| `--venv` | `VLLM_VENV` | `<main checkout>/.venv-vllm` (worktrees reuse it) |

Validation, done before anything starts:
- **TP and GPUs.** TP must be in the model's allowed set. The GPU mask must be non-empty, contain distinct
  indices (normalized to decimal first, so `0,00` is a duplicate and `00,01` is exported as `0,1`), and have exactly TP entries. An *unset* `CUDA_VISIBLE_DEVICES` means 0..TP-1; a set-but-*empty*
  one is an error.
- **Values.** Ports, lengths, utilisation, fps and frames are checked as numbers before they go into the
  command or the JSON. fps and GPU utilisation must be plain decimals (`0.5`, `2`, `2.25`, not `01`, `.5`, `1.`, `1e3`). `--mm-processor-kwargs` must be a JSON object. An option with no value is an error.
- **Before launch** (skipped by `--dry-run`): the GPU indices must exist (`nvidia-smi -L`), and the port must be
  free (`/proc/net/tcp*`), so an old server can't answer the health check.

Extra vLLM args go after `--`. Parallelism flags (`--tensor-parallel-size`/`-tp`, `-pp`, `-dp`, also as
`--flag=value`) and flags the script manages (`--port`, `--host`, `--max-model-len`, `--revision`,
`--gpu-memory-utilization`, `--mm-processor-kwargs`, `--media-io-kwargs`) are refused there. (The old
`EXTRA_ARGS` env var is gone; it was word-split unquoted.)

### wait_healthy.sh

`--model ID` is required. Ready means `/v1/models` parses as JSON and `data[].id` contains exactly ID; an empty
list, another model or a prefix match keeps waiting. With `--pid` (local URL), each poll first checks the PID is
alive (exit 2 if it died). Ready is reported **only once `/proc` confirms** that the PID or a child owns the
listening socket. If another process owns it, the script exits 2; if ownership can't be confirmed yet, it keeps
polling until the timeout. For a remote `--url` (host other than localhost/127.x/::1) the PID/port check is
skipped, and a log line says so. Sleeps and curl timeouts are capped at the time left, so `--timeout` is
honoured. Exit codes: 0 ready, 1 timeout, 2 server died / foreign port owner, 3 bad arguments.

Fixed flags, taken from the reference recipe: `--async-scheduling --enable-chunked-prefill --allowed-local-media-path /
--mm-processor-cache-gb 0 --mm-encoder-tp-mode data --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}'`,
and env `VLLM_USE_FLASHINFER_SAMPLER=0`. At TP>1 the scripts also add `--disable-custom-all-reduce`, because
the 4090 has no P2P. All of these flags were checked against vLLM 0.32.0's own CLI parser.

Sampling: the checkpoint's `generation_config.json` has `do_sample=true, temperature=0.7, top_p=0.8, top_k=20`.
vLLM uses these as **server-side defaults** unless a request overrides them. The evaluation client must send
its sampling parameters explicitly (e.g. `temperature=0`), or the server can be started with `-- --generation-config vllm`.

## Which vLLM, and the evidence

**Cosmos3ForConditionalGeneration is in upstream vLLM from v0.23.0 onward.** We pin **`vllm==0.32.0+cu129`**.

Evidence (all collected without a GPU):

1. **PyPI sdists, grepped `vllm/model_executor/models/registry.py` and the `models/` directory:**
   | version | `cosmos3.py` | registry entry |
   |---|---|---|
   | 0.21.0, 0.22.0, 0.22.1 | – | – |
   | **0.23.0** (2026-06-13) | yes | `Cosmos3ForConditionalGeneration` |
   | 0.24.0 | yes | yes |
   | 0.28.0, 0.34.0 | yes (+ `cosmos3_edge.py`) | `Cosmos3ForConditionalGeneration`, `Cosmos3EdgeForConditionalGeneration` |
2. **Upstream PR:** the history of `cosmos3.py` at v0.23.0 starts with "Add Cosmos3 Reasoner model (#43356)",
   commit `9aa131f9`, 2026-05-28.
3. **Installed venv check** (`setup_vllm_env.sh`, run 2026-11-25): `vllm 0.32.0 | torch 2.13.0+cu129 (cuda 12.9)`,
   `registry Cosmos3ForConditionalGeneration: yes`, `Qwen3VLForConditionalGeneration: yes`.
4. **The docker image** `vllm/vllm-openai:cosmos3@sha256:db0bb920…` (amd64 manifest `sha256:1605bf6a…`), read from
   the registry API config blob without pulling the image: built 2026-07-20 from vLLM commit
   `bfee6a802906cdc4762018d99a26e125067dd6f7`. That commit is a merge of `main` into a feature branch
   (`bastefaniak/fix_cosmos3_edge_checkpoint_weights_filtering`), not a release tag; by date it falls between
   v0.25.1 and v0.26.0. The image uses `CUDA_VERSION=13.0.2`, Python 3.12 and FlashInfer 0.6.14.
5. **What vLLM's Cosmos3 does** (`vllm/model_executor/models/cosmos3.py`, 61 lines): it is
   `class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration)`, i.e. only a weight-name mapper.
   - It loads **only the understanding (reasoner) tower**. It drops `*_moe_gen*`, `add_{q,k,v}_proj`,
     `to_add_out`, `norm_added_{q,k}`, `proj_in/out`, `time_embedder`, the audio/action projections and
     embeddings, and ModelOpt quantizer tensors.
   - **The understanding attention is `to_q/to_k/to_v/to_out` (+ `norm_q/norm_k`)**, renamed to
     `q_proj/k_proj/v_proj/o_proj`. The `add_*` set belongs to the generation tower.
   - `allow_patterns_overrides = ["[tv]*er/*.safetensors"]` means only `transformer/` and `vision_encoder/` shards
     are read, so the VAE and the sound tokenizer are never opened.
   - So on GPU, Cosmos3-Nano is the same size as Qwen3-VL-8B (≈8.8B params, ≈17.6 GB bf16) and Cosmos3-Super the
     same as Qwen3-VL-32B (≈33B, ≈66 GB).
6. **The model card** (HF `nvidia/Cosmos3-Nano`) recommends a different route: `vllm==0.21.0` (cu130) or
   `vllm==0.19.1` (cu128) plus the out-of-tree plugin `vllm-cosmos3` (NVIDIA/cosmos-framework,
   `packages/vllm-cosmos3`, pure Python, `vllm>=0.19`, depends on the monorepo-local `transformers-cosmos3`). The
   plugin registers `Cosmos3ReasonerForConditionalGeneration` and needs
   `--hf-overrides '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}'`. We don't use it: upstream
   ≥0.23 handles the checkpoint's own `Cosmos3ForConditionalGeneration` arch natively, and the reference docker
   image does the same (no override in its flags). If the smoke test shows a mismatch, the plugin is the fallback.

Why 0.32.0 and not the newest release (0.34.0): **driver 550 needs a CUDA 12 build.**
- From 0.23 on, the PyPI wheels are CUDA 13 builds (they depend on `nvidia-cutlass-dsl[cu13]`, and the docker
  image is CUDA 13.0.2). CUDA 13 needs driver ≥ 580, so neither the PyPI wheel nor the image can run here.
- `wheels.vllm.ai/<ver>/cu129/` publishes `+cu129` wheels for 0.23.0 through **0.32.0**. 0.33.0 and 0.34.0 have
  none, and there are no cu126/cu128 vLLM wheels for ≥0.23.
- 0.32.0 pins `torch==2.13.0`, and `download.pytorch.org/whl/cu129` has `torch 2.13.0+cu129`. (torch 2.14, used
  by vLLM 0.34, exists only as cu126/cu130.)
- So 0.32.0+cu129 is the newest vLLM that installs as **pure wheels** (no CUDA source build) for CUDA 12. It also
  includes the later Cosmos3 fixes, e.g. the cosmos3_edge weight-filtering branch the docker image was built from.
  To stay closer to the reference image, use `VLLM_VERSION=0.26.0 serving/setup_vllm_env.sh --force
  --venv /workspace/Cosmos3-Reasoner/.venv-vllm-0.26 --lock /tmp/requirements-vllm-0.26.lock`. cu129 wheels
  exist for it. Pass `--lock` elsewhere so the committed 0.32 lock isn't overwritten.

## Driver 550 / CUDA 12.4: status and risks

Verified on 2026-11-25, with no model loaded:
- `torch.cuda.is_available() == True` on driver 550 with the cu12.9 runtime, compute capability (8, 9). This works
  through CUDA **minor-version compatibility**: a 12.x runtime runs on any driver ≥ 525.
- `vllm/_C_stable_libtorch.abi3.so` contains **sm_89 SASS** (cuobjdump: sm_75/80/86/89/90/100/120). torch ships
  sm_86 SASS, which is binary-compatible with sm_89. vLLM's FlashAttention (`_vllm_fa2_C`) is prebuilt.
- Installed: `transformers 5.19.0`, `flashinfer-python 0.7.0.post1`, `triton 3.7.1`, `nvidia-nccl-cu12 2.29.7`,
  `nvidia-cuda-runtime-cu12 12.9.79`. Venv size **11 GB**.

Remaining risks, to check in the GPU smoke test (t003):
1. **PTX JIT.** Minor-version compatibility does not cover PTX the driver must JIT. If a kernel exists only as
   PTX newer than ISA 8.4, it fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` / "provided PTX was compiled with
   an unsupported toolchain". The vLLM/torch kernels above have SASS. **Triton** compiles with its bundled
   (newer) ptxas straight to cubin, which normally loads. If it doesn't, set
   `TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas`.
2. **FlashInfer JIT** uses the venv's `nvidia-cuda-nvcc-cu12 12.9`. The FlashInfer sampler is off
   (`VLLM_USE_FLASHINFER_SAMPLER=0`), and on sm_89 vLLM picks the FlashAttention backend by default. If a
   FlashInfer path still triggers and fails, force the backend with `-- --attention-backend FLASH_ATTN`.
3. `nvidia-cutlass-dsl-libs-cu13` gets installed as a dependency (CuTe-DSL for Hopper/Blackwell kernels). It should
   never load on sm_89. If an import of it fails at startup, that is the culprit.
4. **Do not enable CUDA forward compatibility** (`VLLM_ENABLE_CUDA_COMPATIBILITY` must stay 0). The cuda-compat
   package doesn't support GeForce cards.
5. **No P2P on the 4090:** `--disable-custom-all-reduce` is set for TP>1, and NCCL falls back to PCIe/SHM.
   If TP init hangs, add `NCCL_P2P_DISABLE=1`.
6. **Shared GPUs:** `--gpu-memory-utilization 0.9` is a fraction of the *total* memory. vLLM aborts if another
   job already holds memory on a chosen GPU. Pick idle GPUs with `--gpus` (`nvidia-smi` first).
7. **Video decoding** goes through vLLM's video loader (OpenCV by default). `num_frames=-1` decodes every frame
   of a long video into host RAM.

If 0.32.0+cu129 hits a blocker that can't be configured around, fall back to `VLLM_VERSION=0.26.0` (or any
0.23–0.31). The fallback after that is the model card's `vllm==0.19.1` cu128 + `vllm-cosmos3` plugin.

## Memory budget: default TP and max-model-len

KV cache per token (bf16) = 2 × layers × KV heads × head_dim × 2 B:
- 8B-class text tower (Cosmos3-Nano, Qwen3-VL-8B: 36 layers, 8 KV heads, 128) = **144 KiB/token**, split across TP.
- 32B-class (Cosmos3-Super, Qwen3-VL-32B: 64 layers, 8 KV heads, 128) = **256 KiB/token**. Because there are
  8 KV heads, TP must be 1, 2, 4 or 8.

4090: 23.99 GiB × 0.9 = **21.6 GiB** per GPU for vLLM. We reserve ≈1.5 GiB for activations, the encoder
profiling peak and CUDA graphs. With `--mm-encoder-tp-mode data`, the ViT (0.58B, 1.1 GiB) is replicated on every GPU.

| model | TP | weights/GPU | KV/GPU | KV pool | **max-model-len** |
|---|---|---|---|---|---|
| Nano / Qwen3-VL-8B | **1** (default) | 16.4 GiB | ≈3.5 GiB | ≈25k tok | **16384** + pixel caps |
| Nano / Qwen3-VL-8B | 2 | ≈8.7 GiB | ≈11 GiB | ≈160k tok | **65536** (128000 also fits, single sequence) |
| Super / Qwen3-VL-32B | **4** (default) | ≈16.4 GiB | ≈3.7 GiB | ≈60k tok | **32768** |
| Super / Qwen3-VL-32B | 8 | ≈8.7 GiB | ≈11 GiB | ≈370k tok | **128000** (= reference) |

These are estimates. vLLM refuses to start if the KV pool is smaller than one `max-model-len` sequence; the log
line `GPU KV cache size: N tokens` gives the real number. If Nano at TP=1 doesn't fit, use `--max-model-len 12288`,
or `--fp8` (see below), or TP=2.

**TP=1 pixel caps (Nano / Qwen3-VL-8B).** The checkpoint's preprocessor budgets are image `longest_edge` =
16,777,216 px (=16,384 visual tokens at 32×32 px/token) and video `longest_edge` = 25,165,824 px over t·h·w
(=12,288 tokens at 32×32×2). Either one alone would fill a 16k context. At TP=1 the scripts pass
`--mm-processor-kwargs '{"images_kwargs": {"max_pixels": 8388608}, "videos_kwargs": {"max_pixels": 12582912}}'`.
That caps one image at ≤8,192 tokens and one video at ≤6,144 tokens. vLLM routes these scoped keys
(`Qwen2VLProcessingInfo._get_vision_size`: `max_pixels` overrides `size.longest_edge`). I checked this on CPU
with vLLM 0.32's processor and the Nano config/processor files (no weights). An 8192×8192 image goes from 16,384
to 8,100 tokens, and a 600-frame 3840×2160 video from 9,600 to 5,400 tokens. This changes the inputs
for very large images/videos compared with the reference (TP≥2 keeps the checkpoint defaults), so report it with
any TP=1 result, or pass `--mm-processor-kwargs none`. The frame cap (`--video-frames`) is left at the
reference's -1: the pixel budget already bounds video tokens, because vLLM shrinks per-frame resolution as the
frame count grows.

**FP8 (option only, off by default).** `--fp8` adds `--quantization fp8 --kv-cache-dtype fp8`. The 4090 (sm_89)
supports FP8. It roughly halves weight memory, which allows Nano at TP=1 with a long context, but it changes
numerics, so it isn't comparable with bf16 / the technical report. Use it only for exploration.

**Recommendation:** for result parity with the report, run Cosmos3-Nano and Qwen3-VL-8B at **TP=2, 65536**
(GPUs 0,1). Use TP=1 / 16384 for quick smoke runs. Run Cosmos3-Super and Qwen3-VL-32B at **TP=4, 32768**, or
**TP=8, 128000** if all 8 GPUs are free. Compare each Cosmos3 model with its Qwen3-VL baseline at the same TP and
max-model-len.

## Downloading weights (not done yet; needs approval)

Repo layout from the HF API at main: Nano `transformer/` 30.35 GB, `vision_encoder/` 1.15 GB, `vae/` 1.41 GB,
`sound_tokenizer/` 1.99 GB. Super `transformer/` 128.04 GB, `vision_encoder/` 1.19 GB, plus the same VAE/sound files.
`download_weights.sh` passes one `--include` per pattern. (`hf download` takes **one** value per `--include`;
extra words become literal FILENAMES, which broke the first version of this script.) The patterns are:
- root files: `config.json generation_config.json chat_template.json preprocessor_config.json
  video_preprocessor_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt`
- directories: `transformer/*` and `vision_encoder/*`

That is enough to load the model. On CPU, `AutoConfig` (→ `Cosmos3OmniConfig`, arch `Cosmos3ForConditionalGeneration`)
and `AutoProcessor` (→ `Qwen3VLProcessor` with chat template) load from exactly these root files, so
`text_tokenizer/`, `vae/`, `sound_tokenizer/`, `assets/`, `images/` and `scheduler/` are skipped.
`hf download --dry-run` at the Nano revision lists **20 files, 31.5 GB**. `transformer/` shards mix both towers
(vLLM drops the generation tensors at load time), so Super is still ≈129 GB. Pin `--revision`
(Nano `45666784803b3a7ad577e0def405f9afcc3bddb2`, Super `73d5746645961a0ec9607123a003bd9c1f8a0282`).

## Docker reference (not usable on this machine)

Two reasons: there is no docker here, and the image is a CUDA 13.0.2 build that wouldn't run on driver 550 anyway.

Image: `vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55`
(vLLM commit `bfee6a80…`, built 2026-07-20, CUDA 13.0.2). The reference flags (H100/H200 class, TP=1):

```bash
docker run --gpus '"device=0"' --ipc=host -p 8080:8080 \
  -v /cache/huggingface:/cache/huggingface -e HF_HOME=/cache/huggingface \
  -v /data0/Cosmos3-Reasoner/LMUData:/data0/Cosmos3-Reasoner/LMUData \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55 \
  nvidia/Cosmos3-Nano \
  --async-scheduling --allowed-local-media-path / --port 8080 \
  --max-model-len 128000 --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill --mm-processor-cache-gb 0 --mm-encoder-tp-mode data \
  --media-io-kwargs '{"video": {"num_frames": -1, "fps": -1}}'
```

Mount the HF cache and LMUData at the **same absolute path** inside the container, because clients send local
media paths that the server opens via `--allowed-local-media-path /`. The native scripts don't need this.
