"""Local keyword retrieval over the scheme chunks.

The agent prefers the hosted Moss index when credentials are configured. This
module is the fallback: it needs no account and no network, so the agent runs
on a laptop with only a Sarvam key, and retrieval can be tested in CI.

It is BM25 over the chunk text - deliberately simple. It ranks well on this
corpus because the questions use the same words as the documents ("documents",
"eligibility", "farmer", "scholarship"). It is not a replacement for an
embedding index on paraphrased or multilingual questions; see the alias table
below for the stopgap that covers the five scheme names.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .chunking import Chunk, chunk_directory

STOPWORDS = set(
    """a an the is are was were be been being of to in on for and or with what
    which who whom how why when where do does did i me my we our you your it its
    this that these those can could should would will shall may might must if
    then than about from by as at into over under any some there their them he
    she his her tell know get got give gives need needs want wants please
    government scheme schemes yojana india indian portal official ministry
    am""".split()
)

# Devanagari words mapped to the English used in the documents. A stopgap: a
# Hindi question should still find PM-KISAN, but this is not a substitute for
# a multilingual embedding index.
ALIASES = {
    "आयुष्मान": "ayushman",
    "स्वास्थ्य": "health",
    "बीमा": "insurance",
    "किसान": "farmer",
    "किसानों": "farmer",
    "सम्मान": "samman",
    "निधि": "nidhi",
    "विश्वकर्मा": "vishwakarma",
    "शिल्पकार": "artisan",
    "कारीगर": "artisan",
    "छात्रवृत्ति": "scholarship",
    "छात्र": "student",
    "कौशल": "skill",
    "प्रशिक्षण": "training",
    "योजना": "scheme",
    "सरकारी": "government",
    "पैसा": "money",
    "पैसे": "money",
    "रुपये": "rupees",
    "दस्तावेज": "documents",
    "दस्तावेज़": "documents",
    "पात्रता": "eligibility",
    "लाभ": "benefit",
}

# Words that mean the same thing to a caller but not to a keyword index.
# Kept deliberately small: the real fix is an embedding index.
EXPANSIONS = {
    "money": " money rupee rupees amount",
    "rupees": " rupee rupees money",
    "give": " give pay paid amount",
    "cost": " cost amount rupee",
    "papers": " paper document",
}

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alias Devanagari, drop stopwords, strip a plural 's'."""
    lowered = text.lower().replace("₹", " rupees ")
    for hindi, english in ALIASES.items():
        if hindi in lowered:
            lowered = lowered.replace(hindi, f" {english} ")
    for word, expansion in EXPANSIONS.items():
        if word in lowered:
            lowered = re.sub(rf"\b{word}\b", expansion, lowered)
    tokens = []
    for token in TOKEN.findall(lowered):
        if len(token) < 2 or token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class LocalIndex:
    """BM25 index over a list of chunks."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [
            tokenize(f"{c.scheme} {c.heading} {c.text}") for c in self.chunks
        ]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = [len(t) for t in self._tokens]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0

        df: Counter = Counter()
        for tokens in self._tokens:
            df.update(set(tokens))
        n = len(self.chunks)
        self._idf = {
            word: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for word, count in df.items()
        }

    # ------------------------------------------------------------- building
    @classmethod
    def from_directory(cls, directory: str | Path) -> "LocalIndex":
        return cls(chunk_directory(directory))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LocalIndex":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk.from_dict(item) for item in raw])

    def _schemes_named_in(self, terms: Sequence[str]) -> List[str]:
        """Schemes the query names unambiguously.

        Only tokens that belong to exactly one scheme count. "kisan" names
        PM-KISAN; "pm" belongs to three schemes and therefore names none of
        them, which is what keeps a PM-KISAN question out of PM Vishwakarma.
        """
        wanted = set(terms)
        owners: dict = {}
        for scheme in {c.scheme for c in self.chunks}:
            for token in set(tokenize(scheme)):
                owners.setdefault(token, set()).add(scheme)
        named = set()
        for token in wanted:
            schemes_for_token = owners.get(token, set())
            if len(schemes_for_token) == 1:
                named |= schemes_for_token
        return sorted(named)

    # ------------------------------------------------------------ searching
    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Return up to `top_k` chunks as (chunk, score), best first.

        With `min_score=None` a relative floor is used: chunks scoring below
        35% of the best match are dropped, which is what stops a question
        about one scheme from dragging in the other four.
        """
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []

        # If the question names a scheme, stay inside it. Without this, a
        # question about PM-KISAN still pulls in the "Benefits" section of
        # every other scheme, because the vocabulary is shared.
        named = self._schemes_named_in(terms)
        candidates = (
            [i for i, c in enumerate(self.chunks) if c.scheme in named]
            if named
            else range(len(self.chunks))
        )

        scored: List[Tuple[float, int]] = []
        for i in candidates:
            tf = self._tf[i]
            length = self._len[i] or 1
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * length / (self._avgdl or 1)
                )
                score += idf * numerator / denominator
            if score > 0:
                scored.append((score, i))

        if not scored:
            return []

        scored.sort(reverse=True)
        floor = min_score if min_score is not None else scored[0][0] * 0.40
        return [(self.chunks[i], score) for score, i in scored[:top_k] if score >= floor]


def search_directory(directory: str | Path, query: str, top_k: int = 4):
    """Convenience wrapper: build an index from a folder and query it once."""
    return LocalIndex.from_directory(directory).search(query, top_k=top_k)
