"""
search.py
---------
Finds candidate web pages for a piece of text.

Three interchangeable backends, all with a free tier:

  duckduckgo  no API key at all       pip install ddgs
  google_cse  100 free queries/day    programmablesearchengine.google.com
  serper      2,500 free queries      serper.dev

Pick one with SEARCH_BACKEND in config.py or the environment.
Every backend returns the same shape: a list of SearchResult.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

import config


@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""
    engine: str = ""


class SearchError(Exception):
    """Raised when a backend is unusable (bad key, missing package, blocked)."""


# ------------------------------------------------------------------ helpers
def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _keep(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    host = _domain(url)
    if any(host == b or host.endswith("." + b) for b in config.BLOCKED_DOMAINS):
        return False
    # Binary files we cannot usefully parse as HTML
    return not url.lower().endswith((".jpg", ".png", ".gif", ".zip", ".mp4", ".mp3"))


# ----------------------------------------------------------------- backends
def _search_duckduckgo(query: str, limit: int) -> list[SearchResult]:
    try:
        from ddgs import DDGS                      # current package name
    except ImportError:
        try:
            from duckduckgo_search import DDGS     # older package name
        except ImportError as exc:
            raise SearchError(
                "DuckDuckGo backend needs a package:  pip install ddgs"
            ) from exc

    out: list[SearchResult] = []
    try:
        with DDGS(timeout=config.SEARCH_TIMEOUT) as ddgs:
            for row in ddgs.text(query, max_results=limit * 2):
                url = row.get("href") or row.get("link") or row.get("url") or ""
                if _keep(url):
                    out.append(
                        SearchResult(
                            url=url,
                            title=row.get("title", ""),
                            snippet=row.get("body") or row.get("snippet", ""),
                            engine="duckduckgo",
                        )
                    )
                if len(out) >= limit:
                    break
    except Exception as exc:
        raise SearchError(f"DuckDuckGo search failed: {exc}") from exc
    return out


def _search_google_cse(query: str, limit: int) -> list[SearchResult]:
    if not (config.GOOGLE_API_KEY and config.GOOGLE_CSE_ID):
        raise SearchError(
            "Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use the Google backend."
        )
    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": config.GOOGLE_API_KEY,
                "cx": config.GOOGLE_CSE_ID,
                "q": query,
                "num": min(limit, 10),
            },
            timeout=config.SEARCH_TIMEOUT,
        )
        if response.status_code == 429:
            raise SearchError("Google daily free quota (100 queries) is used up.")
        response.raise_for_status()
        items = response.json().get("items", [])
    except SearchError:
        raise
    except Exception as exc:
        raise SearchError(f"Google search failed: {exc}") from exc

    return [
        SearchResult(
            url=item.get("link", ""),
            title=item.get("title", ""),
            snippet=item.get("snippet", ""),
            engine="google_cse",
        )
        for item in items
        if _keep(item.get("link", ""))
    ][:limit]


def _search_serper(query: str, limit: int) -> list[SearchResult]:
    if not config.SERPER_API_KEY:
        raise SearchError("Set SERPER_API_KEY to use the Serper backend.")
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": config.SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": min(limit, 10)},
            timeout=config.SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        items = response.json().get("organic", [])
    except Exception as exc:
        raise SearchError(f"Serper search failed: {exc}") from exc

    return [
        SearchResult(
            url=item.get("link", ""),
            title=item.get("title", ""),
            snippet=item.get("snippet", ""),
            engine="serper",
        )
        for item in items
        if _keep(item.get("link", ""))
    ][:limit]


_BACKENDS = {
    "duckduckgo": _search_duckduckgo,
    "google_cse": _search_google_cse,
    "serper": _search_serper,
}


# -------------------------------------------------------------------- front
class WebSearcher:
    """
    Thin wrapper that adds a politeness delay, quoting, in-run caching and
    an automatic fallback to DuckDuckGo when a keyed backend runs dry.
    """

    def __init__(self, backend: str | None = None, results_per_query: int | None = None):
        self.backend = (backend or config.SEARCH_BACKEND).lower()
        if self.backend not in _BACKENDS:
            raise SearchError(
                f"Unknown backend '{self.backend}'. "
                f"Choose one of: {', '.join(_BACKENDS)}"
            )
        self.results_per_query = results_per_query or config.RESULTS_PER_QUERY
        self._cache: dict[str, list[SearchResult]] = {}
        self._last_call = 0.0
        self.queries_made = 0
        self.errors: list[str] = []
        self._consecutive_failures = 0
        self.disabled = False        # set after repeated failures

    def _throttle(self) -> None:
        wait = config.SEARCH_DELAY_SECONDS - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def search(self, text: str, exact_phrase: bool = True) -> list[SearchResult]:
        """
        Search for one chunk. `exact_phrase` wraps the text in quotes, which
        is what makes verbatim copying jump straight to the top of results.
        """
        if self.disabled:
            return []

        words = text.split()[: config.MAX_QUERY_WORDS]
        query = " ".join(words)
        if not query:
            return []
        if exact_phrase:
            query = f'"{query}"'

        if query in self._cache:
            return self._cache[query]

        self._throttle()
        failed = False
        try:
            results = _BACKENDS[self.backend](query, self.results_per_query)
            self.queries_made += 1
        except SearchError as exc:
            self.errors.append(str(exc))
            failed = True
            # Fall back to the keyless backend once, then give up quietly.
            if self.backend != "duckduckgo":
                try:
                    results = _search_duckduckgo(query, self.results_per_query)
                    self.queries_made += 1
                    failed = False
                except SearchError as exc2:
                    self.errors.append(str(exc2))
                    results = []
            else:
                results = []

        if failed:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self.disabled = True
                self.errors.append(
                    "Searching stopped after three failures in a row. "
                    "Check the internet connection, the API key, or install "
                    "the search package."
                )
            return []
        self._consecutive_failures = 0

        # An exact-phrase search that finds nothing is normal for original
        # writing; retry unquoted so paraphrasing still surfaces sources.
        if not results and exact_phrase:
            results = self.search(text, exact_phrase=False)

        self._cache[query] = results
        return results
