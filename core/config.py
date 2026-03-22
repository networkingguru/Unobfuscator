import yaml
from pathlib import Path
from typing import Any

# Project root — two levels up from core/config.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_CACHE_DIR = _PROJECT_ROOT / "pdf_cache"


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f)


def get(cfg: dict, key_path: str, default: Any = None) -> Any:
    """Access nested config values with dot notation: 'matching.min_overlap_chars'"""
    keys = key_path.split(".")
    value = cfg
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            return default
        value = value[k]
    return value
