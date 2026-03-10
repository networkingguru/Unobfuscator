import pytest
import yaml
from pathlib import Path
from core.config import load_config, get


@pytest.fixture
def config_file(tmp_path):
    cfg = {
        "output_dir": "./output",
        "db_path": "./data/test.db",
        "matching": {"min_overlap_chars": 200, "similarity_threshold": 0.70},
        "redaction_markers": ["[REDACTED]", "XXXXXXXXX"],
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return str(p)


def test_load_config_reads_yaml(config_file):
    cfg = load_config(config_file)
    assert cfg["output_dir"] == "./output"
    assert cfg["matching"]["min_overlap_chars"] == 200


def test_get_nested_key(config_file):
    cfg = load_config(config_file)
    assert get(cfg, "matching.min_overlap_chars") == 200


def test_get_returns_default_for_missing_key(config_file):
    cfg = load_config(config_file)
    assert get(cfg, "nonexistent.key", default=99) == 99


def test_load_config_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
