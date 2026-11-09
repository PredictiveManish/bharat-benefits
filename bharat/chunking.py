"""Split the scheme documents into retrievable chunks.

Each scheme file is one document with `## Section` headings. Indexing whole
files meant a "top 3" result was most of the corpus, so retrieval could not
tell one scheme from another and every answer was assembled from everything.
Splitting on the section headings gives ~40 focused chunks instead of 5
undifferentiated blobs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
H2 = re.compile(r"^##\s+(.+)$", re.MULTILINE)
WS = re.compile(r"[ \t]+")
SLUG = re.compile(r"[^a-z0-9]+")
# Sections that are scaffolding for the demo rather than scheme information.
SKIP_HEADING = re.compile(r"demo note", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    """One retrievable section of one scheme document."""

    id: str
    scheme: str
    heading: str
    text: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def _collapse(text: str) -> str:
    """Squash runs of spaces and blank lines but keep the markdown."""
    text = WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scheme_name(text: str, fallback: str) -> str:
    """'# PM-KISAN – Pradhan Mantri Kisan Samman Nidhi' -> 'PM-KISAN'."""
    match = H1.search(text)
    if not match:
        return fallback
    title = match.group(1).strip()
    return re.split(r"\s+[–—-]\s+", title)[0].strip() or fallback


def chunk_markdown(text: str, source: str) -> List[Chunk]:
    """Split one markdown document into chunks, one per `##` section."""
    stem = Path(source).stem
    scheme = scheme_name(text, stem)

    pieces = H2.split(text)
    chunks: List[Chunk] = []

    preamble = _collapse(H1.sub("", pieces[0]))
    if preamble:
        chunks.append(
            Chunk(
                id=f"{stem}#overview",
                scheme=scheme,
                heading="Overview",
                text=preamble,
                source=source,
            )
        )

    for i in range(1, len(pieces), 2):
        heading = pieces[i].strip()
        if SKIP_HEADING.search(heading):
            continue
        body = _collapse(pieces[i + 1]) if i + 1 < len(pieces) else ""
        if not body:
            continue
        slug = SLUG.sub("-", heading.lower()).strip("-") or f"section-{i}"
        chunks.append(
            Chunk(
                id=f"{stem}#{slug}",
                scheme=scheme,
                heading=heading,
                text=body,
                source=source,
            )
        )

    return chunks


def chunk_directory(directory: str | Path) -> List[Chunk]:
    """Chunk every markdown file in a directory, sorted by source then heading."""
    directory = Path(directory)
    chunks: List[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    return chunks


def schemes(chunks: Iterable[Chunk]) -> List[str]:
    """Distinct scheme names, in document order."""
    seen: List[str] = []
    for chunk in chunks:
        if chunk.scheme not in seen:
            seen.append(chunk.scheme)
    return seen
