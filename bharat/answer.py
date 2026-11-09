"""Turn retrieval results into prompt context and a spoken answer.

Two things matter for a benefits assistant:

1. the model must see labelled context, so it can attribute what it says;
2. the answer the user hears must carry the source and the "verify on the
   official portal" line, because a wrong eligibility rule has a real cost.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

DISCLAIMER = "Please verify the details on the official government portal."

NO_MATCH = (
    "I could not find that in the scheme information I have. I can answer "
    "questions about Ayushman Bharat, PM-KISAN, PM Vishwakarma, the National "
    "Scholarship Portal and PMKVY. For anything else, please check the "
    "official government portal."
)

# (label, text) pairs, label being what we want the model to cite.
Entry = Tuple[str, str]

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_EMPHASIS = re.compile(r"[*_`]+")
_MD_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_BLANK = re.compile(r"\n{2,}")


def entries_from_chunks(hits: Sequence) -> List[Entry]:
    """Local retrieval hits -> (label, text) entries."""
    return [(f"{chunk.scheme} — {chunk.heading} ({chunk.source})", chunk.text) for chunk, _ in hits]


def entries_from_moss(docs: Sequence) -> List[Entry]:
    """Moss result documents -> (label, text) entries."""
    entries = []
    for doc in docs:
        metadata = getattr(doc, "metadata", None) or {}
        source = metadata.get("source", getattr(doc, "id", "unknown"))
        entries.append((str(source), getattr(doc, "text", "") or ""))
    return entries


def build_context(entries: Iterable[Entry]) -> str:
    """Labelled context block for the prompt."""
    blocks = [f"[Source: {label}]\n{text}".strip() for label, text in entries if text]
    if not blocks:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(blocks)


def source_names(entries: Iterable[Entry]) -> List[str]:
    """Distinct human-readable source labels, in order of appearance."""
    names: List[str] = []
    for label, _ in entries:
        name = label.split(" (")[0].split(" — ")[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def strip_markdown(text: str) -> str:
    """Flatten markdown so the answer can be spoken."""
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _MD_EMPHASIS.sub("", text)
    text = _BLANK.sub("\n", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def with_citations(answer: str, names: Sequence[str]) -> str:
    """Append the source and the verification reminder, without duplicating them."""
    text = strip_markdown(answer).strip()
    if names and "source:" not in text.lower():
        text = f"{text} Source: {', '.join(names)}."
    if DISCLAIMER.lower() not in text.lower():
        text = f"{text} {DISCLAIMER}"
    return text.strip()
