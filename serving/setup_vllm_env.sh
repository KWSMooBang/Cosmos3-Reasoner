#!/usr/bin/env bash
# Build the native (no-docker) vLLM venv for serving Cosmos3 / Qwen3-VL on this
# machine (RTX 4090, driver 550 = CUDA <= 12.4).  Pure wheel install, no CUDA builds.
# See serving/README.md for why these versions were chosen.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [--venv DIR] [--vllm-version X.Y.Z] [--cuda cu129] [--python 3.12] [--force] [--check-only]

Creates a separate uv venv with vLLM + torch built for CUDA 12.x, then checks that
Cosmos3ForConditionalGeneration is in the vLLM model registry (CPU only, no GPU work).

Options / env vars:
  --venv DIR            VLLM_VENV      (default: <repo>/.venv-vllm)
  --vllm-version X      VLLM_VERSION   (default: 0.32.0; Cosmos3 support exists in >= 0.23.0)
  --cuda TAG            VLLM_CUDA      (default: cu129; wheels.vllm.ai has no cu126/cu128 builds for >= 0.23)
  --python V            VLLM_PYTHON    (default: 3.12, the same as the upstream docker image)
  --force               delete and rebuild an existing venv
  --check-only          skip the install and only run the registry/version check
  -h, --help            show this help
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-${REPO_ROOT%%/.claude/worktrees/*}/.venv-vllm}"  # worktrees share the main checkout venv
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
VLLM_CUDA="${VLLM_CUDA:-cu129}"
VLLM_PYTHON="${VLLM_PYTHON:-3.12}"
FORCE=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VLLM_VENV="$2"; shift 2 ;;
    --vllm-version) VLLM_VERSION="$2"; shift 2 ;;
    --cuda) VLLM_CUDA="$2"; shift 2 ;;
    --python) VLLM_PYTHON="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "uv not found" >&2; exit 1; }
PY="$VLLM_VENV/bin/python"

if [[ $CHECK_ONLY -eq 0 ]]; then
  if [[ -d "$VLLM_VENV" && $FORCE -eq 1 ]]; then rm -rf "$VLLM_VENV"; fi
  [[ -x "$PY" ]] || uv venv --python "$VLLM_PYTHON" --seed "$VLLM_VENV"

  # vLLM's CUDA-12 wheel lives on wheels.vllm.ai (the PyPI wheel is a CUDA 13 build,
  # and CUDA 13 does not run on driver 550). --torch-backend makes uv take
  # torch/torchvision/torchaudio from download.pytorch.org/whl/$VLLM_CUDA.
  uv pip install --python "$PY" \
    --torch-backend "$VLLM_CUDA" \
    --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${VLLM_CUDA}" \
    --index-strategy unsafe-best-match \
    "vllm==${VLLM_VERSION}+${VLLM_CUDA}" \
    openai
fi

# Registry / version check. CPU only: imports vLLM and reads the registry, no CUDA context.
CUDA_VISIBLE_DEVICES="" "$PY" - <<'EOF'
import importlib.metadata as md
import torch, vllm
from vllm import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
need = ["Cosmos3ForConditionalGeneration", "Qwen3VLForConditionalGeneration"]
print(f"vllm {vllm.__version__} | torch {torch.__version__} (cuda {torch.version.cuda})")
for p in ("transformers", "flashinfer-python", "triton", "nvidia-nccl-cu12", "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime"):
    try:
        print(f"  {p} {md.version(p)}")
    except md.PackageNotFoundError:
        pass
missing = [a for a in need if a not in archs]
for a in need:
    print(f"  registry {a}: {'yes' if a in archs else 'NO'}")
if not torch.version.cuda or not torch.version.cuda.startswith("12."):
    raise SystemExit(f"torch is built for CUDA {torch.version.cuda}; driver 550 needs a CUDA 12.x build")
if missing:
    raise SystemExit(f"missing architectures: {missing}")
print("OK")
EOF
echo "venv: $VLLM_VENV ($(du -sh "$VLLM_VENV" | cut -f1))"
