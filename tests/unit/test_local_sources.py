from pathlib import Path

from llm_engineering.application.local_sources import (
    chunk_text,
    extract_source_text,
    load_local_source_documents,
    normalize_text,
)


def test_normalize_text_collapses_spaces_and_blank_lines() -> None:
    assert normalize_text("hello   world\n\n\nsecond\tline") == "hello world\n\nsecond line"


def test_chunk_text_preserves_small_paragraph_groups() -> None:
    chunks = chunk_text("one\n\ntwo\n\nthree", chunk_size=10, overlap=2)

    assert chunks == ["one\n\ntwo", "three"]


def test_extract_source_text_reads_markdown(tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_text("# Title\n\nBody")

    document = extract_source_text(source)

    assert document.source_type == "md"
    assert document.text == "# Title\n\nBody"


def test_load_local_source_documents_creates_missing_dir(tmp_path: Path) -> None:
    source_dir = tmp_path / "missing"

    assert load_local_source_documents(source_dir) == []
    assert source_dir.exists()
