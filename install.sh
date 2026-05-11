#!/usr/bin/env bash
set -e

# 1. Clear stale artifacts
rm -rf mamba_install/build
rm -f uv.lock

# 2. Create the virtual environment explicitly
uv venv
source .venv/bin/activate

# 3. Pre-seed build dependencies using --extra-index-url to keep standard PyPI access
uv pip install setuptools wheel packaging "torch>=2.4.0" --extra-index-url https://download.pytorch.org/whl/cu124

# 4. Sync the rest of the project
uv sync --extra dev