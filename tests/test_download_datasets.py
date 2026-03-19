import json
import os
import pytest
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from download_datasets import (
    DATASETS,
    get_disk_usage,
    check_disk_space,
    get_remote_size,
    download_with_resume,
    extract_and_cleanup,
    write_provenance,
    load_provenance,
)


def test_datasets_contains_expected_entries():
    """All 8 missing datasets should be defined."""
    ids = [d["id"] for d in DATASETS]
    assert ids == [3, 4, 5, 6, 7, 8, 9, 12]


def test_datasets_have_required_fields():
    for d in DATASETS:
        assert "id" in d
        assert "url" in d
        assert "source_type" in d
        assert "source_label" in d


def test_check_disk_space_passes_when_plenty_of_room():
    """With 500GB free on a 1TB disk, 1GB download should be fine."""
    with patch("download_datasets.get_disk_usage") as mock_du:
        mock_du.return_value = MagicMock(total=1_000_000_000_000, free=500_000_000_000)
        assert check_disk_space("/tmp", download_size=1_000_000_000) is True


def test_check_disk_space_fails_when_would_breach_10_percent():
    """With 105GB free on a 1TB disk, 20GB download would leave <10%."""
    with patch("download_datasets.get_disk_usage") as mock_du:
        mock_du.return_value = MagicMock(total=1_000_000_000_000, free=105_000_000_000)
        assert check_disk_space("/tmp", download_size=20_000_000_000) is False


def test_write_and_load_provenance(tmp_path):
    prov_path = tmp_path / "provenance.json"
    write_provenance(str(prov_path), 3, {
        "source_url": "https://example.com/DataSet%203.zip",
        "source_type": "archive.org",
        "source_label": "Internet Archive mirror — DOJ original unavailable",
    })
    data = load_provenance(str(prov_path))
    assert "3" in data
    assert data["3"]["source_type"] == "archive.org"


def test_write_provenance_preserves_existing(tmp_path):
    prov_path = tmp_path / "provenance.json"
    write_provenance(str(prov_path), 3, {"source_type": "archive.org"})
    write_provenance(str(prov_path), 4, {"source_type": "archive.org"})
    data = load_provenance(str(prov_path))
    assert "3" in data
    assert "4" in data


def test_extract_and_cleanup(tmp_path):
    """Create a test zip, extract it, verify zip is deleted."""
    zip_path = tmp_path / "test.zip"
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()

    # Create a zip with a test PDF
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test.pdf", b"fake pdf content")

    extract_and_cleanup(str(zip_path), str(extract_dir))

    assert (extract_dir / "test.pdf").exists()
    assert not zip_path.exists()  # zip should be deleted
