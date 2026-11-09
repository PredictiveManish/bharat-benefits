"""Tests for splitting scheme documents into chunks."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bharat.chunking import chunk_directory, chunk_markdown, schemes

SCHEMES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "schemes"
)


def test_every_scheme_is_represented():
    chunks = chunk_directory(SCHEMES_DIR)
    names = schemes(chunks)
    assert len(names) == 5
    assert "PM-KISAN" in names
    assert "Ayushman Bharat" in names


def test_chunks_are_sections_not_whole_files():
    """The old index had five documents; retrieval could not discriminate."""
    chunks = chunk_directory(SCHEMES_DIR)
    assert len(chunks) > 25
    assert max(len(c.text) for c in chunks) < 2000


def test_chunk_ids_are_unique():
    chunks = chunk_directory(SCHEMES_DIR)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunks_carry_their_source():
    for chunk in chunk_directory(SCHEMES_DIR):
        assert chunk.source.endswith(".md")
        assert chunk.scheme
        assert chunk.heading
        assert chunk.text.strip()


def test_demo_scaffolding_is_not_indexed():
    headings = {c.heading.lower() for c in chunk_directory(SCHEMES_DIR)}
    assert not any("demo note" in h for h in headings)


def test_scheme_name_comes_from_the_title():
    text = "# PM-KISAN – Pradhan Mantri Kisan Samman Nidhi\n\n## Purpose\nmoney\n"
    chunks = chunk_markdown(text, "pm-kisan.md")
    assert chunks[0].scheme == "PM-KISAN"


def test_markdown_splits_on_headings():
    text = "# Thing\n\nintro line\n\n## One\nalpha\n\n## Two\nbeta\n"
    chunks = chunk_markdown(text, "thing.md")
    assert [c.heading for c in chunks] == ["Overview", "One", "Two"]
    assert chunks[1].text == "alpha"
