#!/usr/bin/env bash
set -euo pipefail

# This is intentionally the only required command. It installs a pinned isolated
# environment, then executes all non-interactive stages in the pre-specified plan.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
STUDY_CONFIG="${STUDY_CONFIG:-configs/study.yaml}"
DEVICE="${DEVICE:-auto}"

if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON_BIN" ] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "A Python 3.10--3.12 interpreter was not found. Set PYTHON_BIN if it is installed under another name." >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))'; then
  echo "This artifact requires Python 3.10--3.12." >&2
  exit 2
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# Do not rely on activation: Git Bash on Windows creates .venv/Scripts, whereas
# POSIX Python creates .venv/bin. Calling the interpreter directly makes the one
# documented command portable across both layouts.
if [ -x .venv/bin/python ]; then
  VENV_PYTHON=".venv/bin/python"
elif [ -x .venv/Scripts/python.exe ]; then
  VENV_PYTHON=".venv/Scripts/python.exe"
else
  echo "The virtual environment was created but its Python executable was not found." >&2
  exit 2
fi

"$VENV_PYTHON" -m pip install --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
"$VENV_PYTHON" -m pip install --requirement requirements.txt
"$VENV_PYTHON" -m pip install --editable .

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME="$REPO_ROOT/.cache/torch"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"

"$VENV_PYTHON" -m qc_curation.cli all --config "$STUDY_CONFIG" --device "$DEVICE"

echo
echo "Experiment complete. Fill manuscript tags using paper/generated/RESULTS_TO_PASTE.md."
