"""
similarity.py
-------------
The mathematics of the project.

Three measures are combined, because each one alone is easy to fool:

1. TF-IDF cosine similarity
   Text becomes a vector weighted by term frequency x inverse document
   frequency, so rare words count more than "the". Cosine of the angle
   between two vectors gives similarity in [0, 1].
   Strength: survives reordering and light paraphrasing.
   Weakness: two texts about the same topic look similar even when
   independently written.

2. N-gram containment (Broder's containment coefficient)
   containment = |ngrams(A) & ngrams(B)| / |ngrams(A)|
   With n = 5, sharing many 5-word sequences is very strong evidence of
   copying, because word order has to match exactly.
   Strength: almost no false positives on verbatim copying.
   Weakness: defeated by heavy rewording.

3. Longest common word sequence
   Not used for scoring - it extracts the actual overlapping phrase so the
   report can show the evidence to a human.

Final score = 0.5 * cosine + 0.5 * containment  (weights live in config.py)

Uses scikit-learn when installed; otherwise a pure-Python implementation
of the same formulae keeps the tool working.
"""

from __future__ import annotations

import math
from collections import Counter

import config
from detector.preprocess import tokenize, word_ngrams, normalise_for_compare

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
    SKLEARN = True
except ImportError:                                     # pragma: no cover
    SKLEARN = False


# ------------------------------------------------------- pure-python TF-IDF
def _fallback_tfidf(texts: list[str]) -> list[dict[str, float]]:
    """L2-normalised tf-idf vectors as sparse dicts. Same formula sklearn uses."""
    docs = [tokenize(t) for t in texts]
    n_docs = len(docs) or 1

    df: Counter[str] = Counter()
    for tokens in docs:
        df.update(set(tokens))

    vectors: list[dict[str, float]] = []
    for tokens in docs:
        tf = Counter(tokens)
        vec: dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0   # smoothed
            vec[term] = count * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({k: v / norm for k, v in vec.items()})
    return vectors


def _fallback_cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


# ------------------------------------------------------------- public API
class TfidfComparer:
    """
    Fits one shared vocabulary over the document chunks *and* every source
    chunk, then produces a cosine similarity matrix between the two groups.

    Fitting both groups together is important: idf must be computed over the
    same corpus, otherwise the two sets of vectors are not comparable.
    """

    def __init__(self, doc_texts: list[str], source_texts: list[str]):
        self.n_doc = len(doc_texts)
        self.n_src = len(source_texts)
        corpus = [normalise_for_compare(t) for t in doc_texts + source_texts]

        if SKLEARN and corpus:
            self._vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),      # unigrams + bigrams add word-order signal
                sublinear_tf=True,       # 1 + log(tf) damps repeated words
                min_df=1,
                norm="l2",
                strip_accents="unicode",
            )
            self._matrix = self._vectorizer.fit_transform(corpus)
        else:
            self._matrix = _fallback_tfidf(corpus)

    def cosine_matrix(self) -> list[list[float]]:
        """matrix[i][j] = similarity(document chunk i, source chunk j)."""
        if self.n_doc == 0 or self.n_src == 0:
            return [[] for _ in range(self.n_doc)]

        if SKLEARN:
            doc_part = self._matrix[: self.n_doc]
            src_part = self._matrix[self.n_doc:]
            return _sk_cosine(doc_part, src_part).tolist()

        doc_part = self._matrix[: self.n_doc]
        src_part = self._matrix[self.n_doc:]
        return [[_fallback_cosine(d, s) for s in src_part] for d in doc_part]


def containment(doc_tokens: list[str], source_ngrams: set) -> float:
    """Fraction of the document's n-grams that also appear in the source."""
    grams = word_ngrams(doc_tokens)
    if not grams:
        return 0.0
    return len(grams & source_ngrams) / len(grams)


def jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Reported as a secondary statistic."""
    a, b = set(tokens_a), set(tokens_b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def longest_common_phrase(tokens_a: list[str], tokens_b: list[str]) -> list[str]:
    """
    Longest run of identical consecutive words. Classic dynamic-programming
    longest-common-substring, run over words instead of characters.
    Memory is kept to two rows, so long pages stay cheap.
    """
    if not tokens_a or not tokens_b:
        return []

    previous = [0] * (len(tokens_b) + 1)
    best_len = 0
    best_end = 0
    for i in range(1, len(tokens_a) + 1):
        current = [0] * (len(tokens_b) + 1)
        token_a = tokens_a[i - 1]
        for j in range(1, len(tokens_b) + 1):
            if token_a == tokens_b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best_len:
                    best_len = current[j]
                    best_end = i
        previous = current
    return tokens_a[best_end - best_len: best_end]


def combined_score(cosine: float, contain: float) -> float:
    return (
        config.TFIDF_WEIGHT * cosine
        + config.CONTAINMENT_WEIGHT * contain
    )
