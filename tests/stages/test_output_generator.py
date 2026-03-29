import pytest
import fitz
import json
from pathlib import Path
from core.db import (
    init_db, get_connection, upsert_document,
    create_match_group, add_group_member, upsert_merge_result
)
from stages.output_generator import YELLOW, GREEN, build_output_path, generate_output_pdf


@pytest.fixture
def conn(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    c = get_connection(path)
    yield c
    c.close()


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return str(d)


def seed_group(conn, base_text, merged_text, recovered_count, segments=None,
               source="doj", batch="VOL00001"):
    for doc_id, text in [(1, base_text), (2, "donor text")]:
        upsert_document(conn, {
            "id": doc_id, "source": source, "release_batch": batch,
            "original_filename": f"doc{doc_id}.pdf", "page_count": 1,
            "size_bytes": 500, "description": "", "extracted_text": text
        })
    conn.commit()
    g = create_match_group(conn)
    add_group_member(conn, g, 1, 1.0)
    add_group_member(conn, g, 2, 0.9)
    upsert_merge_result(conn, g, merged_text, recovered_count=recovered_count,
                        total_redacted=max(recovered_count, 1), source_doc_ids=[1, 2],
                        recovered_segments=segments or [])
    conn.commit()
    return g


def test_build_output_path_uses_source_batch_and_doc_id(output_dir):
    path = build_output_path(output_dir, source="doj", batch="VOL00008", doc_id=12345)
    assert path.endswith("12345_merged.pdf")
    assert "doj" in path
    assert "VOL00008" in path


def test_build_output_path_creates_parent_directories(output_dir):
    path = build_output_path(output_dir, source="house_oversight",
                             batch="DataSet11", doc_id=99)
    assert Path(path).parent.exists()


def test_generate_output_pdf_creates_file(conn, output_dir):
    segments = [{"text": "Bill Clinton", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="The passenger list included [REDACTED] on the flight.",
                   merged_text="The passenger list included Bill Clinton on the flight.",
                   recovered_count=1, segments=segments)
    path = generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                               redaction_markers=["[REDACTED]"])
    assert path is not None
    assert Path(path).exists()
    assert Path(path).suffix == ".pdf"


def test_generate_output_pdf_contains_recovered_text(conn, output_dir):
    segments = [{"text": "Donald Trump", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="Attendees: [REDACTED]. Location: Palm Beach.",
                   merged_text="Attendees: Donald Trump. Location: Palm Beach.",
                   recovered_count=1, segments=segments)
    path = generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                               redaction_markers=["[REDACTED]"])
    doc = fitz.open(path)
    full_text = "".join(page.get_text() for page in doc)
    assert "Donald Trump" in full_text
    doc.close()


def test_generate_output_pdf_footnote_has_sources_and_recovery_method(conn, output_dir):
    segments = [{"text": "recovered", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="Text with [REDACTED] inside.",
                   merged_text="Text with recovered inside.",
                   recovered_count=1, segments=segments)
    path = generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                               redaction_markers=["[REDACTED]"])
    doc = fitz.open(path)
    last_page_text = doc[-1].get_text()
    assert "SOURCE DOCUMENT" in last_page_text
    assert "Unobfuscator" in last_page_text
    assert "Method:" in last_page_text
    doc.close()


def test_generate_output_pdf_returns_none_when_no_recoveries(conn, output_dir):
    g = seed_group(conn,
                   base_text="Clean text.",
                   merged_text="Clean text.",
                   recovered_count=0)
    path = generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                               redaction_markers=["[REDACTED]"])
    assert path is None


def test_generate_output_pdf_marks_output_generated(conn, output_dir):
    segments = [{"text": "the President", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="The [REDACTED] attended the meeting.",
                   merged_text="The the President attended the meeting.",
                   recovered_count=1, segments=segments)
    generate_output_pdf(conn, group_id=g, output_dir=output_dir,
                        redaction_markers=["[REDACTED]"])
    conn.commit()
    row = conn.execute(
        "SELECT output_generated, output_path FROM merge_results WHERE group_id = ?", (g,)
    ).fetchone()
    assert row["output_generated"] == 1
    assert row["output_path"] is not None
    assert row["output_path"].endswith("_merged.pdf")


def test_metadata_page_includes_provenance_notice(conn, output_dir, tmp_path):
    """Documents from non-DOJ sources should show provenance on metadata page."""
    segments = [{"text": "secret name", "source_doc_id": 2, "stage": "merge"}]
    g = seed_group(conn,
                   base_text="Some text with [REDACTED] in it.",
                   merged_text="Some text with secret name in it.",
                   recovered_count=1, segments=segments,
                   batch="VOL00008")

    # Write provenance file
    prov_path = str(tmp_path / "provenance.json")
    with open(prov_path, "w") as f:
        json.dump({"8": {
            "source_label": "Source: Community mirror (GeekenDev) — DOJ original unavailable",
            "source_type": "community-mirror:geeken",
        }}, f)

    path = generate_output_pdf(conn, g, output_dir, ["[REDACTED]"],
                               provenance_path=prov_path)
    assert path is not None

    # Search all pages for provenance text (metadata may span pages)
    doc = fitz.open(path)
    full_text = "".join(page.get_text() for page in doc)
    assert "Community mirror (GeekenDev)" in full_text
    assert "DOJ original unavailable" in full_text
    doc.close()
