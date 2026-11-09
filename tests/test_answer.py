"""Tests for the spoken answer: sources and the verification line."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bharat.answer import (
    DISCLAIMER,
    NO_MATCH,
    build_context,
    source_names,
    strip_markdown,
    with_citations,
)


def test_context_labels_every_block():
    context = build_context([("PM-KISAN — Benefits (pm-kisan.md)", "₹6,000 per year")])
    assert "[Source: PM-KISAN — Benefits (pm-kisan.md)]" in context
    assert "₹6,000" in context


def test_empty_context_says_so():
    assert "No relevant information" in build_context([])


def test_source_names_are_short_and_distinct():
    entries = [
        ("PM-KISAN — Benefits (pm-kisan.md)", "a"),
        ("PM-KISAN — Eligibility Summary (pm-kisan.md)", "b"),
        ("PMKVY — Purpose (pmkvy-skill-india.md)", "c"),
    ]
    assert source_names(entries) == ["PM-KISAN", "PMKVY"]


def test_markdown_is_removed_before_speaking():
    spoken = strip_markdown("## Benefits\n- **₹6,000** per year\n- Paid in [instalments](http://x)")
    assert "**" not in spoken
    assert "#" not in spoken
    assert spoken.startswith("Benefits")
    assert "instalments" in spoken


def test_citations_are_appended_once():
    once = with_citations("PM-KISAN pays 6000 rupees a year.", ["PM-KISAN"])
    twice = with_citations(once, ["PM-KISAN"])
    assert once == twice
    assert once.count("Source:") == 1
    assert once.count(DISCLAIMER) == 1


def test_disclaimer_is_always_present():
    answer = with_citations("Some answer.", ["PMKVY"])
    assert answer.endswith(DISCLAIMER)


def test_no_match_names_the_schemes_it_covers():
    assert "PM-KISAN" in NO_MATCH
    assert "official government portal" in NO_MATCH
