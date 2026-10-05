"""
preprocess.py
-------------
Cleaning, sentence splitting, chunking and query building.

Why chunks?  Comparing two whole documents gives one blurry number.
Comparing ~45-word chunks tells you *which sentences* were copied and
*where they came from*, which is what a plagiarism report needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config

# Sentence boundary: ".", "!" or "?" followed by whitespace + capital/quote.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9']+")
_NON_TEXT = re.compile(r"[^A-Za-z0-9\s'\-.,;:!?()\"]+")


@dataclass
class Chunk:
    """One comparable slice of a document."""
    index: int
    text: str
    start_char: int
    end_char: int
    word_count: int
    tokens: list[str] = field(default_factory=list)

    @property
    def query(self) -> str:
        """The phrase we hand to the search engine."""
        words = self.text.split()[: config.MAX_QUERY_WORDS]
        return " ".join(words)


# --------------------------------------------------------------- tokenising
def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    """Lowercase word tokens. Stopwords kept by default (they matter for n-grams)."""
    tokens = _WORD.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in config.STOPWORDS]
    return tokens


def normalise_for_compare(text: str) -> str:
    """Aggressive normalisation used before vectorising."""
    text = _NON_TEXT.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------- chunking
def chunk_document(text: str, target_words: int | None = None) -> list[Chunk]:
    """
    Group whole sentences into chunks of about `target_words` words.
    Sentence boundaries are never broken, so every chunk reads naturally
    and searches well.
    """
    target_words = target_words or config.CHUNK_WORDS
    sentences = split_sentences(text)

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0
    cursor = 0  # running position for start/end char offsets

    def flush() -> None:
        nonlocal buffer, buffer_words, cursor
        if not buffer:
            return
        body = " ".join(buffer)
        words = len(body.split())
        if words >= config.CHUNK_MIN_WORDS:
            start = text.find(buffer[0], cursor)
            start = cursor if start == -1 else start
            end = start + len(body)
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=body,
                    start_char=start,
                    end_char=end,
                    word_count=words,
                    tokens=tokenize(body),
                )
            )
            cursor = max(cursor, end)
        buffer = []
        buffer_words = 0

    for sentence in sentences:
        words = len(sentence.split())
        # A single very long sentence becomes its own chunk.
        if words >= target_words * 1.6:
            flush()
            buffer = [sentence]
            buffer_words = words
            flush()
            continue

        # Close the current chunk rather than let it overshoot. Oversized
        # chunks mix copied and original sentences together, which drags a
        # genuine match below the threshold.
        if buffer and buffer_words + words > target_words * 1.3:
            flush()

        buffer.append(sentence)
        buffer_words += words
        if buffer_words >= target_words:
            flush()

    flush()
    return chunks


def word_ngrams(tokens: list[str], n: int | None = None) -> set[tuple[str, ...]]:
    """Set of word n-grams, used for the containment score."""
    n = n or config.NGRAM_SIZE
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def select_query_chunks(chunks: list[Chunk], limit: int | None = None) -> list[Chunk]:
    """
    We cannot hit the search engine once per chunk on a long document,
    so pick an informative, evenly spread subset.

    Informative = many non-stopword tokens (generic filler sentences like
    "This is very important." return useless search results).
    """
    limit = limit or config.MAX_CHUNKS_TO_SEARCH
    usable = [c for c in chunks if len(set(c.tokens) - config.STOPWORDS) >= 8]
    if not usable:
        usable = list(chunks)
    if len(usable) <= limit:
        return usable

    step = len(usable) / limit
    picked = [usable[int(i * step)] for i in range(limit)]

    # De-duplicate while preserving document order.
    seen: set[int] = set()
    unique: list[Chunk] = []
    for chunk in picked:
        if chunk.index not in seen:
            seen.add(chunk.index)
            unique.append(chunk)
    return unique
