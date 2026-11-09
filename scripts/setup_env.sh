#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Creates the virtual environment ./.venv with uv and installs everything
# pinned in uv.lock (Flower, PyTorch, MONAI, ...) plus the Flower app in
# fl_app/ (editable mode). uv downloads a suitable Python if needed.
#
# Usage (from the repository root):
#   bash scripts/setup_env.sh              # default PyTorch build (CUDA on Linux)
#   bash scripts/setup_env.sh --cpu        # CPU-only PyTorch (much smaller download)
#   PYTHON=3.13 bash scripts/setup_env.sh  # choose the Python version (3.11-3.13)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_EXTRA="cuda"

for arg in "$@"; do
    case "$arg" in
        --cpu) TORCH_EXTRA="cpu" ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# --- 1. Check that uv is installed ----------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install it with:" >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# --- 2. Create .venv and install the locked dependencies ------------------
cd "$REPO_ROOT"
echo ">>> Installing the locked dependencies (PyTorch build: $TORCH_EXTRA)"
uv sync --locked --extra "$TORCH_EXTRA" ${PYTHON:+--python "$PYTHON"}

# --- 3. Sanity check -------------------------------------------------------
uv run --no-sync python - <<'EOF'
import flwr, torch, torchvision, monai, numpy
print("\n=== Environment ready ===")
print(f"flwr        {flwr.__version__}")
print(f"torch       {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"torchvision {torchvision.__version__}")
print(f"monai       {monai.__version__}")
print(f"numpy       {numpy.__version__}")
EOF

echo
echo "Activate the environment in every new terminal with:"
echo "    source $REPO_ROOT/.venv/bin/activate"
