"""
engine.py
---------
Runs the whole pipeline and returns a Report object.

    document -> text -> chunks -> web search -> fetch sources
             -> source chunks -> TF-IDF cosine + n-gram containment
             -> per-chunk verdict -> overall similarity score
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import config
from detector import similarity
from detector.extractor import extract_text
from detector.fetcher import WebPage, fetch_many
from detector.preprocess import (
    Chunk,
    chunk_document,
    select_query_chunks,
    tokenize,
    word_ngrams,
)
from detector.search import WebSearcher, SearchError


# ------------------------------------------------------------------ results
@dataclass
class ChunkMatch:
    chunk_index: int
    chunk_text: str
    word_count: int
    score: float
    cosine: float
    containment: float
    source_url: str
    source_title: str
    source_domain: str
    matched_excerpt: str
    verdict: str                    # "copied" | "similar" | "original"


@dataclass
class SourceSummary:
    url: str
    title: str
    domain: str
    chunks_matched: int
    words_matched: int
    best_score: float
    percent_of_document: float


@dataclass
class Report:
    filename: str
    checked_at: str
    total_words: int
    total_chunks: int
    similarity_percent: float          # headline number
    copied_words: int
    matches: list[ChunkMatch] = field(default_factory=list)
    sources: list[SourceSummary] = field(default_factory=list)
    queries_made: int = 0
    pages_fetched: int = 0
    duration_seconds: float = 0.0
    backend: str = ""
    warnings: list[str] = field(default_factory=list)
    document_text: str = ""

    @property
    def verdict(self) -> str:
        if self.similarity_percent >= 40:
            return "High plagiarism risk"
        if self.similarity_percent >= 15:
            return "Moderate plagiarism risk"
        if self.similarity_percent >= 5:
            return "Low plagiarism risk"
        return "No significant plagiarism"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["verdict"] = self.verdict
        data.pop("document_text", None)
        return data


# ------------------------------------------------------------------- engine
class PlagiarismEngine:
    """
    progress_callback(stage: str, current: int, total: int, message: str)
    lets a UI show a live progress bar. It is optional.
    """

    def __init__(self, backend: str | None = None, progress_callback=None):
        self.backend = backend or config.SEARCH_BACKEND
        self.progress = progress_callback or (lambda *a, **k: None)

    # ------------------------------------------------------------ public
    def check_file(self, path: str | Path) -> Report:
        path = Path(path)
        self.progress("reading", 0, 1, f"Reading {path.name}")
        text = extract_text(path)
        return self.check_text(text, filename=path.name)

    def check_text(self, text: str, filename: str = "pasted text") -> Report:
        started = time.time()
        warnings: list[str] = []

        chunks = chunk_document(text)
        total_words = len(text.split())
        if not chunks:
            return Report(
                filename=filename,
                checked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                total_words=total_words,
                total_chunks=0,
                similarity_percent=0.0,
                copied_words=0,
                warnings=["The document was too short to analyse."],
                backend=self.backend,
            )

        # 1. search the web ------------------------------------------------
        queries = select_query_chunks(chunks)
        urls: list[str] = []
        try:
            searcher = WebSearcher(self.backend)
        except SearchError as exc:
            raise RuntimeError(str(exc)) from exc

        for i, chunk in enumerate(queries, start=1):
            self.progress("searching", i, len(queries),
                          f"Searching the web for passage {i} of {len(queries)}")
            for result in searcher.search(chunk.query):
                urls.append(result.url)

        warnings.extend(dict.fromkeys(searcher.errors))
        if not urls:
            warnings.append(
                "No web results came back. Check your internet connection, "
                "or try another SEARCH_BACKEND in config.py."
            )

        # 2. download the candidate pages ---------------------------------
        def fetch_progress(done, total, url):
            self.progress("fetching", done, total, f"Reading {url[:70]}")

        pages = fetch_many(urls, progress=fetch_progress) if urls else []
        if urls and not pages:
            warnings.append("Every candidate page failed to download.")

        # 3. compare -------------------------------------------------------
        self.progress("comparing", 0, 1, "Comparing text with the sources")
        matches, sources = self._compare(chunks, pages)

        copied_words = sum(m.word_count for m in matches if m.verdict == "copied")
        percent = round(100 * copied_words / max(total_words, 1), 2)

        report = Report(
            filename=filename,
            checked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_words=total_words,
            total_chunks=len(chunks),
            similarity_percent=percent,
            copied_words=copied_words,
            matches=matches,
            sources=sources,
            queries_made=searcher.queries_made,
            pages_fetched=len(pages),
            duration_seconds=round(time.time() - started, 1),
            backend=self.backend,
            warnings=warnings,
            document_text=text,
        )
        self.progress("done", 1, 1, "Finished")
        return report

    def check_against_local(self, path: str | Path,
                            corpus_dir: str | Path) -> Report:
        """Offline mode: compare one file against a folder of other documents."""
        started = time.time()
        path = Path(path)
        text = extract_text(path)
        chunks = chunk_document(text)

        pages: list[WebPage] = []
        for candidate in sorted(Path(corpus_dir).iterdir()):
            if candidate.suffix.lower() not in config.ALLOWED_EXTENSIONS:
                continue
            if candidate.resolve() == path.resolve():
                continue
            try:
                pages.append(
                    WebPage(url=f"file://{candidate}", title=candidate.name,
                            text=extract_text(candidate))
                )
            except Exception as exc:
                print(f"  skipped {candidate.name}: {exc}")

        matches, sources = self._compare(chunks, pages)
        copied_words = sum(m.word_count for m in matches if m.verdict == "copied")
        total_words = len(text.split())
        return Report(
            filename=path.name,
            checked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_words=total_words,
            total_chunks=len(chunks),
            similarity_percent=round(100 * copied_words / max(total_words, 1), 2),
            copied_words=copied_words,
            matches=matches,
            sources=sources,
            pages_fetched=len(pages),
            duration_seconds=round(time.time() - started, 1),
            backend="local corpus",
            document_text=text,
        )

    # ----------------------------------------------------------- internal
    def _compare(self, chunks: list[Chunk], pages: list[WebPage]):
        """Score every document chunk against every source chunk."""
        if not pages:
            return (
                [self._empty_match(c) for c in chunks],
                [],
            )

        # Break each page into chunks and remember which page each came from.
        source_texts: list[str] = []
        owner: list[int] = []                 # source chunk index -> page index
        page_ngrams: list[set] = []
        for p_index, page in enumerate(pages):
            page_ngrams.append(word_ngrams(tokenize(page.text)))
            for source_chunk in chunk_document(page.text):
                source_texts.append(source_chunk.text)
                owner.append(p_index)

        if not source_texts:
            return ([self._empty_match(c) for c in chunks], [])

        comparer = similarity.TfidfComparer([c.text for c in chunks], source_texts)
        cosines = comparer.cosine_matrix()

        matches: list[ChunkMatch] = []
        per_page = {
            i: {"chunks": 0, "words": 0, "best": 0.0} for i in range(len(pages))
        }

        for chunk, row in zip(chunks, cosines):
            best = None
            # Only the strongest few candidates are worth the expensive
            # containment + LCS work.
            ranked = sorted(range(len(row)), key=lambda j: row[j], reverse=True)[:8]
            for j in ranked:
                cosine = float(row[j])
                if cosine < 0.10:
                    continue
                p_index = owner[j]
                contain = similarity.containment(chunk.tokens, page_ngrams[p_index])
                score = similarity.combined_score(cosine, contain)
                if best is None or score > best[0]:
                    best = (score, cosine, contain, p_index, source_texts[j])

            if best is None:
                matches.append(self._empty_match(chunk))
                continue

            score, cosine, contain, p_index, src_text = best
            page = pages[p_index]
            verdict = (
                "copied" if score >= config.MATCH_THRESHOLD
                else "similar" if score >= config.REPORT_THRESHOLD
                else "original"
            )
            excerpt = " ".join(
                similarity.longest_common_phrase(chunk.tokens, tokenize(src_text))
            )

            matches.append(
                ChunkMatch(
                    chunk_index=chunk.index,
                    chunk_text=chunk.text,
                    word_count=chunk.word_count,
                    score=round(score, 3),
                    cosine=round(cosine, 3),
                    containment=round(contain, 3),
                    source_url=page.url if verdict != "original" else "",
                    source_title=page.title if verdict != "original" else "",
                    source_domain=page.domain if verdict != "original" else "",
                    matched_excerpt=excerpt if verdict != "original" else "",
                    verdict=verdict,
                )
            )

            if verdict == "copied":
                stats = per_page[p_index]
                stats["chunks"] += 1
                stats["words"] += chunk.word_count
                stats["best"] = max(stats["best"], score)

        total_words = sum(c.word_count for c in chunks) or 1
        sources = [
            SourceSummary(
                url=pages[i].url,
                title=pages[i].title or pages[i].url,
                domain=pages[i].domain,
                chunks_matched=s["chunks"],
                words_matched=s["words"],
                best_score=round(s["best"], 3),
                percent_of_document=round(100 * s["words"] / total_words, 2),
            )
            for i, s in per_page.items()
            if s["chunks"] > 0
        ]
        sources.sort(key=lambda s: s.percent_of_document, reverse=True)
        return matches, sources

    @staticmethod
    def _empty_match(chunk: Chunk) -> ChunkMatch:
        return ChunkMatch(
            chunk_index=chunk.index,
            chunk_text=chunk.text,
            word_count=chunk.word_count,
            score=0.0,
            cosine=0.0,
            containment=0.0,
            source_url="",
            source_title="",
            source_domain="",
            matched_excerpt="",
            verdict="original",
        )
