"""Where everything lives -- the one module that knows the folder layout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
FIGURES = ROOT / "figures"
DATASETS = ROOT / "datasets"
CACHE = DATASETS / "cache"
MODELS = DATASETS / "models"
RESULTS = DATASETS / "results"
