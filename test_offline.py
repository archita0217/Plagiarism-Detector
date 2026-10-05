"""
test_offline.py
---------------
Proves the detection logic works without touching the internet, which is
handy for a viva demonstration when the campus Wi-Fi is unreliable.

    python test_offline.py

It checks two sample essays against samples/source_article.txt:
the copied one should score high, the original one near zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import config
from detector import report as reporting
from detector.engine import PlagiarismEngine
from detector.preprocess import chunk_document, tokenize, word_ngrams
from detector.similarity import (
    TfidfComparer, containment, jaccard, longest_common_phrase,
)

SAMPLES = config.SAMPLE_DIR


def test_similarity_measures() -> bool:
    a = "the first flush device throws away the dirty opening minutes of a storm"
    b = "a first flush device throws away the dirty opening minutes of a storm"
    c = "our captain waits for a batsman to hit two boundaries before switching"

    cos = TfidfComparer([a], [b, c]).cosine_matrix()[0]
    contain_ab = containment(tokenize(a), word_ngrams(tokenize(b)))
    contain_ac = containment(tokenize(a), word_ngrams(tokenize(c)))
    phrase = " ".join(longest_common_phrase(tokenize(a), tokenize(b)))

    print("Similarity measures")
    print(f"  cosine(a, near-identical) = {cos[0]:.3f}")
    print(f"  cosine(a, unrelated)      = {cos[1]:.3f}")
    print(f"  containment(a, near)      = {contain_ab:.3f}")
    print(f"  containment(a, unrelated) = {contain_ac:.3f}")
    print(f"  jaccard(a, near)          = {jaccard(tokenize(a), tokenize(b)):.3f}")
    print(f"  longest shared phrase     = '{phrase}'")

    passed = cos[0] > 0.6 and cos[0] > cos[1] and contain_ab > 0.6 and contain_ac < 0.1
    print("  result:", "pass" if passed else "FAIL")
    return passed


def test_chunking() -> bool:
    text = (SAMPLES / "source_article.txt").read_text(encoding="utf-8")
    chunks = chunk_document(text)
    sizes = [c.word_count for c in chunks]
    print("\nChunking")
    print(f"  {len(chunks)} chunks, sizes {min(sizes)}-{max(sizes)} words")
    passed = len(chunks) > 3 and all(s >= config.CHUNK_MIN_WORDS for s in sizes)
    print("  result:", "pass" if passed else "FAIL")
    return passed


def test_detection() -> bool:
    engine = PlagiarismEngine()
    corpus = SAMPLES

    print("\nDetection against a local corpus")
    copied = engine.check_against_local(SAMPLES / "copied_essay.txt", corpus)
    original = engine.check_against_local(SAMPLES / "original_essay.txt", corpus)

    print(f"  copied_essay.txt   -> {copied.similarity_percent:g}% ({copied.verdict})")
    print(f"  original_essay.txt -> {original.similarity_percent:g}% ({original.verdict})")

    out = reporting.save_html(copied, config.REPORT_DIR / "offline_demo.html")
    reporting.save_json(copied, config.REPORT_DIR / "offline_demo.json")
    print(f"  demo report written to {out}")

    passed = copied.similarity_percent > 25 and original.similarity_percent < 10
    print("  result:", "pass" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    results = [test_similarity_measures(), test_chunking(), test_detection()]
    print("\n" + ("All checks passed." if all(results) else "Some checks FAILED."))
    sys.exit(0 if all(results) else 1)
