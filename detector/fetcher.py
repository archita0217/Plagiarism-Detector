"""
fetcher.py
----------
Downloads candidate pages and reduces them to readable body text.

Features that matter for a real tool:
  * on-disk cache, so re-running a check costs nothing
  * robots.txt is honoured
  * parallel downloads with a thread pool
  * boilerplate (nav, scripts, footers) stripped before comparison
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

import config


@dataclass
class WebPage:
    url: str
    title: str
    text: str
    ok: bool = True
    error: str = ""

    @property
    def domain(self) -> str:
        parsed = urlparse(self.url)
        if parsed.scheme == "file" or not parsed.netloc:
            # Local corpus mode: the file name is the useful label.
            return self.title or self.url.rsplit("/", 1)[-1]
        host = parsed.netloc.lower()
        return host[4:] if host.startswith("www.") else host


# -------------------------------------------------------------------- cache
def _cache_path(url: str):
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return config.CACHE_DIR / f"{key}.json"


def _read_cache(url: str) -> WebPage | None:
    path = _cache_path(url)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > config.CACHE_TTL_HOURS:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WebPage(**data)
    except Exception:
        return None


def _write_cache(page: WebPage) -> None:
    try:
        _cache_path(page.url).write_text(
            json.dumps(asdict(page), ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


# ------------------------------------------------------------- robots.txt
_robots_cache: dict[str, RobotFileParser | None] = {}


def _allowed_by_robots(url: str) -> bool:
    if not config.RESPECT_ROBOTS_TXT:
        return True
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root not in _robots_cache:
        parser = RobotFileParser()
        parser.set_url(root + "/robots.txt")
        try:
            parser.read()
        except Exception:
            parser = None            # unreachable robots.txt -> allow
        _robots_cache[root] = parser
    parser = _robots_cache[root]
    if parser is None:
        return True
    try:
        return parser.can_fetch(config.USER_AGENT, url)
    except Exception:
        return True


# ------------------------------------------------------------ html -> text
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript|svg|nav|header|footer|form)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\u00a0]+")
_NL = re.compile(r"\n{3,}")


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, body_text). Uses BeautifulSoup when available."""
    title = ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        for tag in soup(
            ["script", "style", "noscript", "nav", "header", "footer",
             "form", "aside", "svg", "iframe"]
        ):
            tag.decompose()

        # Prefer the main article container if the page marks one.
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text(separator="\n")
    except ImportError:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        title = _TAG.sub("", match.group(1)).strip() if match else ""
        cleaned = _SCRIPT_STYLE.sub(" ", html)
        text = _TAG.sub(" ", cleaned)

    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&")
        .replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop menu-ish one-or-two word lines; keep real sentences.
    lines = [ln for ln in lines if len(ln.split()) > 3]
    text = _NL.sub("\n\n", _WS.sub(" ", "\n".join(lines))).strip()
    return title, text[: config.MAX_PAGE_CHARS]


# ----------------------------------------------------------------- fetching
def fetch_page(url: str, session: requests.Session | None = None) -> WebPage:
    cached = _read_cache(url)
    if cached is not None:
        return cached

    if not _allowed_by_robots(url):
        return WebPage(url, "", "", ok=False, error="Blocked by robots.txt")

    session = session or requests
    try:
        response = session.get(
            url,
            headers={
                "User-Agent": config.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=config.FETCH_TIMEOUT,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type and "text" not in content_type:
            return WebPage(url, "", "", ok=False,
                           error=f"Not a text page ({content_type})")

        response.encoding = response.encoding or response.apparent_encoding
        title, text = html_to_text(response.text)
        if len(text.split()) < 40:
            return WebPage(url, title, text, ok=False, error="Too little text")

        page = WebPage(url=url, title=title or url, text=text)
        _write_cache(page)
        return page
    except Exception as exc:
        return WebPage(url, "", "", ok=False, error=str(exc)[:200])


def fetch_many(urls: list[str], progress=None) -> list[WebPage]:
    """Download a list of URLs in parallel. `progress` is an optional callback."""
    urls = list(dict.fromkeys(urls))[: config.MAX_SOURCES_TO_FETCH]
    pages: list[WebPage] = []
    done = 0

    session = requests.Session()
    with ThreadPoolExecutor(max_workers=config.FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_page, u, session): u for u in urls}
        for future in as_completed(futures):
            page = future.result()
            done += 1
            if progress:
                progress(done, len(urls), page.url)
            if page.ok:
                pages.append(page)
    session.close()
    return pages
