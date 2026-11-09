# 1. Environment setup

Every project ends up with dependency conflicts sooner or later (one library wants
`numpy<2`, another `numpy>=2`...). To avoid breaking your system Python, install
everything in a **virtual environment** (venv) that lives in the project folder.
The project is managed with [uv](https://docs.astral.sh/uv/): the dependencies are declared in
[`pyproject.toml`](../pyproject.toml) and their exact versions are pinned in `uv.lock`.

## Requirements

* **Linux or macOS.** On **Windows, use WSL2**: open PowerShell as administrator, run
  `wsl --install`, reboot, open the "Ubuntu" app and follow this guide inside it.
* **uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh` (check with `uv --version`).
* **Python 3.11, 3.12 or 3.13** (Flower needs ≥ 3.11). You do not need to install it yourself:
  uv downloads Python 3.12 (see `.python-version`) if it is not on your machine.
* About 2 GB of free disk space for the CPU build, or 6 GB or more for the CUDA build.

## One-command install

From the repository root:

```bash
bash scripts/setup_env.sh --cpu      # CPU-only PyTorch: small download, works everywhere
# or
bash scripts/setup_env.sh            # default PyTorch build (includes CUDA on Linux)
```

To use another Python version than 3.12:

```bash
PYTHON=3.13 bash scripts/setup_env.sh --cpu
```

The script runs `uv sync`, which:

1. creates `.venv/` in the repository (with a uv-managed Python if needed),
2. installs PyTorch (CPU or default build, chosen with the `cpu` / `cuda` extra),
3. installs everything pinned in `uv.lock`, plus `pytest` and `ruff`,
4. installs the Flower app in `fl_app/` in editable mode (it is a uv workspace member), so that
   `import aitdm_fl` works anywhere (handy for your centralised baseline or a notebook),

and then prints the versions it installed:

```text
=== Environment ready ===
flwr        1.41.0
torch       2.14.1+cpu  (CUDA available: False)
torchvision 0.29.1+cpu
monai       1.6.1
numpy       2.5.2
```

## Using the environment

Activate it in **every new terminal**:

```bash
source .venv/bin/activate        # your prompt now starts with (.venv)
deactivate                       # leave it
```

The `scripts/*.sh` files activate `.venv` for you, so you can run them directly.
Without activating, prefix a command with `uv run --no-sync`, e.g.
`uv run --no-sync python data_splitting/split_data.py ...` (`--no-sync` keeps the PyTorch
build you chose; a plain `uv run` would re-sync without the `cpu` extra).

## Where the code you will edit lives

The Flower app is the Python package `aitdm_fl`, inside the `fl_app/` folder of the repository.
These are the files you will change during the project (paths from the repository root,
`AITDM_FLOWER_PROJECT_TUTORIAL/`):

| File | Full path | What is in it |
|---|---|---|
| `task.py` | `AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/task.py` | The model (`build_model()`, `build_mlp()`), data loading, `train()` and `evaluate()`. It contains no Flower code, so you can reuse it for your centralised baseline |
| `server_app.py` | `AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/server_app.py` | The **server**: aggregation strategy (`build_strategy()`), differential privacy, global evaluation, and saving `final_model.pt` + `metrics.json` |
| `client_app.py` | `AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/client_app.py` | The **client**: local training / evaluation on one partition, plus the optional attacker |
| `pyproject.toml` | `AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/pyproject.toml` | App settings and the default run configuration (rounds, learning rate, strategy, ...) |

To print the **absolute** paths on your machine (they depend on where you cloned the
repository), run this with the venv activated:

```bash
python -c "import aitdm_fl.task as t, aitdm_fl.server_app as s; print(t.__file__); print(s.__file__)"
```

```text
/home/<you>/AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/task.py
/home/<you>/AITDM_FLOWER_PROJECT_TUTORIAL/fl_app/aitdm_fl/server_app.py
```

Because the app is installed in editable mode ([step 5 above](#one-command-install)), Python always uses these files
directly. Your edits take effect at the next run, without reinstalling. If the command prints a
path inside `.venv/lib/.../site-packages/` instead, the app was not installed in editable mode:
run `uv sync --extra cpu` (or `--extra cuda`) from the repository root.

## What is installed and why

| Package | Why |
|---|---|
| `flwr[simulation,dp]` | Flower framework; `[simulation]` adds Ray for the Simulation Runtime, `[dp]` adds `dp-accounting` (privacy budget ε) |
| `flwr-datasets[vision]` | Optional: Flower's own dataset partitioners (Hugging Face datasets) |
| `torch`, `torchvision` | Deep learning |
| `monai` | Medical imaging networks, transforms and losses |
| `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm` | Data handling, metrics (F1, balanced accuracy), plots |
| `grad-cam` | Grad-CAM / Grad-CAM++ / Score-CAM explanations for Stage 2 (pulls in OpenCV), see [tutorial 5](05_gradcam.md) |

Need another library? Add it with uv, which updates `pyproject.toml` and `uv.lock`
(commit both so your teammates get the same version):

```bash
uv add --no-sync shap==0.48.0         # updates pyproject.toml and uv.lock
uv sync --extra cpu                   # installs it (keep cpu or cuda, as in your setup)
```

## Doing it by hand (what the script does)

```bash
uv sync --extra cpu                   # or --extra cuda for the default (CUDA) PyTorch build
source .venv/bin/activate
```

`uv sync --no-dev` skips `pytest` and `ruff`. Need a `requirements.txt` for another tool?
`uv export --extra cpu --no-hashes > requirements.txt` writes one from the lock file.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ERROR: 'uv' not found` | `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new terminal |
| `The lockfile at uv.lock needs to be updated` | Someone edited `pyproject.toml` without `uv lock`: run `uv lock` and commit `uv.lock` |
| Very large download / disk full | Use `--cpu`; delete `.venv/` and start again |
| `CUDA available: False` but you have a GPU | Re-create the venv without `--cpu`; check `nvidia-smi` works |
| University cluster (FEP) | Run the server and all clients **on the same node and queue**, or use the node's real IP instead of `127.0.0.1`.<br><br>Otherwise, pointing to 0.0.0.0:8080 or 127.0.0.1:8080 is going to point to different actual IPs and this will mean that they are not communicating (server is waiting for the clients and clients give an error of "no server to talk to") |
