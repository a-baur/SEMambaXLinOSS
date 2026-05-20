import sys
from pathlib import Path

# pyproject.toml sets pythonpath = ["src"], but models/ lives at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
