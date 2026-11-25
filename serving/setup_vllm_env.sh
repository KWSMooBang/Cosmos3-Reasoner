#!/usr/bin/env bash
# Build the native (no-docker) vLLM venv for serving Cosmos3 / Qwen3-VL on this machine
# (RTX 4090, driver 550 = CUDA <= 12.4). Wheels only (--only-binary :all:), no CUDA builds.
# See serving/README.md for why these versions were chosen.
set -euo pipefail

SERVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SERVING_DIR/.." && pwd)"
MAIN_ROOT="${REPO_ROOT%%/.claude/worktrees/*}"          # worktrees share the main checkout's venv
SAFE_ROOT="/workspace/Cosmos3-Reasoner"                  # --force only deletes venvs under here

usage() {
  cat <<EOF
Usage: $(basename "$0") [--from-lock] [--venv DIR] [--vllm-version X.Y.Z] [--cuda cu129] [--python 3.12]
                        [--lock FILE] [--force] [--check-only]

Creates a separate uv venv with vLLM + torch built for CUDA 12.x, then checks it (CPU only, no GPU work):
exact Python / vllm / torch versions and that Cosmos3ForConditionalGeneration is in the vLLM registry.

Modes:
  (default)             resolve vllm==\$VLLM_VERSION+\$VLLM_CUDA fresh, then write the resolved set to --lock
  --from-lock           install exactly the packages in --lock (uv pip sync); reproduces the committed env
  --check-only          no install; only run the checks

Options / env vars:
  --venv DIR            VLLM_VENV      (default: $MAIN_ROOT/.venv-vllm)
  --vllm-version X      VLLM_VERSION   (default: 0.32.0; Cosmos3 support exists in >= 0.23.0)
  --cuda TAG            VLLM_CUDA      (default: cu129; wheels.vllm.ai has no cu126/cu128 builds for >= 0.23)
  --python V            VLLM_PYTHON    (default: 3.12, the same as the upstream docker image)
  --lock FILE           VLLM_LOCK      (default: $SERVING_DIR/requirements-vllm.lock)
  --force               delete and rebuild an existing venv (only a real venv, i.e. with pyvenv.cfg,
                        under $SAFE_ROOT)
  -h, --help            show this help
EOF
}
die() { echo "error: $*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && "$2" != --* ]] || { echo "error: $1 needs a value" >&2; usage >&2; exit 2; }; }

VLLM_VENV="${VLLM_VENV:-$MAIN_ROOT/.venv-vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
VLLM_CUDA="${VLLM_CUDA:-cu129}"
VLLM_PYTHON="${VLLM_PYTHON:-3.12}"
VLLM_LOCK="${VLLM_LOCK:-$SERVING_DIR/requirements-vllm.lock}"
FORCE=0; MODE=resolve

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) need_value "$@"; VLLM_VENV="$2"; shift 2 ;;
    --vllm-version) need_value "$@"; VLLM_VERSION="$2"; shift 2 ;;
    --cuda) need_value "$@"; VLLM_CUDA="$2"; shift 2 ;;
    --python) need_value "$@"; VLLM_PYTHON="$2"; shift 2 ;;
    --lock) need_value "$@"; VLLM_LOCK="$2"; shift 2 ;;
    --from-lock) MODE=lock; shift ;;
    --check-only) MODE=check; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$VLLM_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "bad --vllm-version '$VLLM_VERSION'"
[[ "$VLLM_CUDA" =~ ^cu12[0-9]$ ]] || die "--cuda must be a CUDA 12 tag like cu129 (driver 550 cannot run CUDA 13)"
[[ "$VLLM_PYTHON" =~ ^3\.[0-9]+$ ]] || die "bad --python '$VLLM_PYTHON' (want e.g. 3.12)"
command -v uv >/dev/null || die "uv not found"

# Symlink check on the raw path, before realpath resolves it: otherwise --force would delete the target.
if [[ $FORCE == 1 && -L "${VLLM_VENV%/}" ]]; then
  die "--force refuses to work through a symlink: $VLLM_VENV -> $(readlink "${VLLM_VENV%/}")"
fi
VLLM_VENV="$(realpath -m "$VLLM_VENV")"
PY="$VLLM_VENV/bin/python"

if [[ $MODE == lock ]]; then
  [[ -f "$VLLM_LOCK" ]] || die "lock file not found: $VLLM_LOCK"
  # The lock pins the version; the checks below must agree with it.
  pinned=$(grep -E '^vllm==' "$VLLM_LOCK" | head -1 | cut -d= -f3)
  [[ "$pinned" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\+(cu12[0-9])$ ]] || die "no vllm==X.Y.Z+cu12N pin in $VLLM_LOCK"
  VLLM_VERSION="${BASH_REMATCH[1]}"; VLLM_CUDA="${BASH_REMATCH[2]}"
  lock_py=$(sed -nE 's/^# python: ([0-9]+\.[0-9]+).*/\1/p' "$VLLM_LOCK" | head -1)
  [[ -n "$lock_py" ]] && VLLM_PYTHON="$lock_py"
fi

if [[ $MODE != check ]]; then
  if [[ $FORCE == 1 && -e "$VLLM_VENV" ]]; then
    [[ "$VLLM_VENV" == "$SAFE_ROOT"/* ]] || die "--force refuses to delete $VLLM_VENV: not under $SAFE_ROOT"
    [[ -f "$VLLM_VENV/pyvenv.cfg" ]] || die "--force refuses to delete $VLLM_VENV: no pyvenv.cfg, not a venv"
    echo "[setup] removing venv $VLLM_VENV"
    rm -rf -- "$VLLM_VENV"
  fi
  [[ -x "$PY" ]] || uv venv --python "$VLLM_PYTHON" --seed "$VLLM_VENV"

  # vLLM's CUDA-12 wheel lives on wheels.vllm.ai (the PyPI wheel is a CUDA 13 build, which driver 550
  # cannot run). --torch-backend makes uv take torch/torchvision/torchaudio/triton from PyTorch's
  # $VLLM_CUDA index (a plain --extra-index-url to download.pytorch.org does not resolve the +cu129 pins).
  idx=(
    --torch-backend "$VLLM_CUDA"
    --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/${VLLM_CUDA}"
    --index-strategy unsafe-best-match
    --only-binary :all:
  )
  if [[ $MODE == lock ]]; then
    uv pip sync --python "$PY" "${idx[@]}" "$VLLM_LOCK"
  else
    # setuptools: torch imports it at runtime (cpp_extension) but only the venv seed provides it.
    uv pip install --python "$PY" "${idx[@]}" "vllm==${VLLM_VERSION}+${VLLM_CUDA}" openai setuptools
  fi
fi

# Checks. CPU only: imports vLLM and reads the registry, no CUDA context.
CUDA_VISIBLE_DEVICES="" "$PY" -I - "$VLLM_PYTHON" "$VLLM_VERSION" "$VLLM_CUDA" <<'EOF'
import importlib.metadata as md, re, sys
want_py, want_vllm, cuda_tag = sys.argv[1], sys.argv[2], sys.argv[3]
errors = []
py = "%d.%d" % sys.version_info[:2]
if py != want_py:
    errors.append(f"python {py} != {want_py}")
vllm_v = md.version("vllm")
if vllm_v != f"{want_vllm}+{cuda_tag}":
    errors.append(f"vllm {vllm_v} != {want_vllm}+{cuda_tag}")
# torch must be exactly the version vLLM pins, built for the same CUDA tag (e.g. 2.13.0+cu129).
pin = next((m.group(1) for r in (md.requires("vllm") or [])
            if (m := re.match(r"torch\s*==\s*([0-9.]+)\s*$", r.split(";")[0].strip()))), None)
torch_v = md.version("torch")
if pin is None or torch_v != f"{pin}+{cuda_tag}":
    errors.append(f"torch {torch_v} != {pin}+{cuda_tag} (vllm's pin)")
import torch
want_cuda = f"{cuda_tag[2:4]}.{cuda_tag[4:]}"          # cu129 -> 12.9
if torch.version.cuda != want_cuda:
    errors.append(f"torch CUDA {torch.version.cuda} != {want_cuda}")
from vllm import ModelRegistry
archs = set(ModelRegistry.get_supported_archs())
# Import the `vllm serve` CLI path too (it pulls in renderers / compilation / pynccl -> torch cpp_extension),
# so a venv that imports vllm but cannot start the server fails here, not at serve time.
try:
    from vllm.entrypoints.launchers.cli_args import make_arg_parser  # noqa: F401
    import vllm.entrypoints.openai.api_server  # noqa: F401
except Exception as e:  # noqa: BLE001
    errors.append(f"serve entrypoint import failed: {type(e).__name__}: {e}")
print(f"python {sys.version.split()[0]} | vllm {vllm_v} | torch {torch_v} (cuda {torch.version.cuda})")
for p in ("transformers", "flashinfer-python", "triton", "nvidia-nccl-cu12", "nvidia-cuda-runtime-cu12"):
    try:
        print(f"  {p} {md.version(p)}")
    except md.PackageNotFoundError:
        pass
for a in ("Cosmos3ForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
    ok = a in archs
    print(f"  registry {a}: {'yes' if ok else 'NO'}")
    if not ok:
        errors.append(f"{a} missing from the vLLM registry")
if errors:
    raise SystemExit("CHECK FAILED: " + "; ".join(errors))
print("OK")
EOF

if [[ $MODE == resolve ]]; then
  {
    echo "# Resolved by serving/setup_vllm_env.sh on $(date -u +%F). Reinstall exactly: serving/setup_vllm_env.sh --from-lock"
    echo "# python: $("$PY" -I -c 'import sys; print(sys.version.split()[0])')"
    echo "# indexes: --torch-backend ${VLLM_CUDA} + https://wheels.vllm.ai/${VLLM_VERSION}/${VLLM_CUDA} + PyPI (unsafe-best-match, --only-binary :all:)"
    # Keep everything, including setuptools: torch.utils.cpp_extension imports it at runtime on
    # Python 3.12, and `uv pip sync` removes whatever the lock does not list.
    uv pip freeze --python "$PY" --exclude-editable
  } > "$VLLM_LOCK"
  echo "[setup] wrote $VLLM_LOCK ($(grep -cvE '^#' "$VLLM_LOCK") packages)"
fi
echo "venv: $VLLM_VENV ($(du -sh "$VLLM_VENV" | cut -f1))"
