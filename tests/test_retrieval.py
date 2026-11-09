"""Tests for local retrieval ranking.

These are the questions the README advertises; each must land on the right
scheme, otherwise the answer is grounded in the wrong document.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from bharat.retrieval import LocalIndex, tokenize

SCHEMES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "schemes"
)


@pytest.fixture(scope="module")
def index():
    return LocalIndex.from_directory(SCHEMES_DIR)


def top_scheme(index, question):
    hits = index.search(question, top_k=3)
    return hits[0][0].scheme if hits else None


def test_documents_question_lands_on_the_documents_section(index):
    hits = index.search("What documents are needed for Ayushman Bharat?", top_k=3)
    assert hits[0][0].scheme == "Ayushman Bharat"
    assert "document" in hits[0][0].heading.lower()


def test_farmer_money_question_lands_on_pm_kisan(index):
    assert top_scheme(index, "How much money does PM-KISAN give and how often?") == "PM-KISAN"


def test_carpenter_question_lands_on_vishwakarma(index):
    assert top_scheme(index, "I am a carpenter. Is there any government scheme for me?") == (
        "PM Vishwakarma Yojana"
    )


def test_student_question_lands_on_scholarships(index):
    assert top_scheme(index, "I am a student looking for scholarship support.") == (
        "National Scholarship Portal (NSP)"
    )


def test_vocational_training_lands_on_pmkvy(index):
    assert top_scheme(index, "I want free vocational training. What should I do?") == "PMKVY"


def test_naming_a_scheme_keeps_results_inside_it(index):
    """A PM-KISAN question must not drag in other schemes' benefit sections."""
    hits = index.search("How much money does PM-KISAN give and how often?", top_k=4)
    assert hits
    assert {chunk.scheme for chunk, _ in hits} == {"PM-KISAN"}


def test_hindi_question_reaches_the_right_scheme(index):
    assert top_scheme(index, "किसान को कितना पैसा मिलता है?") == "PM-KISAN"


def test_off_topic_question_returns_nothing(index):
    assert index.search("What is the weather in Mumbai tomorrow?") == []


def test_scores_are_ordered(index):
    hits = index.search("documents required for PM Vishwakarma", top_k=3)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_index_round_trips_through_json(index, tmp_path):
    path = tmp_path / "index.json"
    index.save(path)
    reloaded = LocalIndex.load(path)
    question = "documents needed for Ayushman Bharat"
    assert [c.id for c, _ in reloaded.search(question, top_k=3)] == [
        c.id for c, _ in index.search(question, top_k=3)
    ]


def test_tokenizer_strips_stopwords_and_plurals():
    tokens = tokenize("What documents are needed for the farmers")
    assert "document" in tokens
    assert "farmer" in tokens
    assert "the" not in tokens and "for" not in tokens
