"""
config.py
---------
Every tunable setting for the plagiarism detector lives here.
Change values here instead of hunting through the code.

API keys are read from environment variables (or a .env file) so that
you never commit secrets. None of them are mandatory: the default
search backend (DuckDuckGo) needs no key at all.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv not installed - not fatal
    pass


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CACHE_DIR = BASE_DIR / "cache"
REPORT_DIR = BASE_DIR / "reports"
SAMPLE_DIR = BASE_DIR / "samples"

for _d in (UPLOAD_DIR, CACHE_DIR, REPORT_DIR, SAMPLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- documents
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".rtf"}
MAX_UPLOAD_MB = 10


# ---------------------------------------------------------------- chunking
# A "chunk" is the unit we search the web with and score.
CHUNK_WORDS = 35          # target words per chunk (sentences are never split)
CHUNK_MIN_WORDS = 12      # ignore chunks shorter than this
MAX_QUERY_WORDS = 28      # search engines ignore very long queries


# ---------------------------------------------------------------- searching
# Which backend to use: "duckduckgo" (free, no key), "google_cse", "serper".
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "duckduckgo")

# Free 100 queries/day: https://programmablesearchengine.google.com/
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Free 2,500 queries on signup: https://serper.dev/
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

MAX_CHUNKS_TO_SEARCH = 25     # cap on web queries per document
RESULTS_PER_QUERY = 5         # candidate URLs per chunk
SEARCH_DELAY_SECONDS = 1.2    # politeness delay between queries
SEARCH_TIMEOUT = 20


# ---------------------------------------------------------------- fetching
MAX_SOURCES_TO_FETCH = 40     # cap on pages downloaded per document
FETCH_TIMEOUT = 15
FETCH_WORKERS = 6             # parallel downloads
MAX_PAGE_CHARS = 200_000      # truncate huge pages
CACHE_TTL_HOURS = 72
USER_AGENT = (
    "Mozilla/5.0 (compatible; PlagiarismDetector/1.0; academic project; "
    "+https://example.edu/plagiarism-project)"
)
RESPECT_ROBOTS_TXT = True

# Pages that are never worth downloading
BLOCKED_DOMAINS = {
    "facebook.com", "instagram.com", "x.com", "twitter.com",
    "pinterest.com", "linkedin.com", "youtube.com", "tiktok.com",
}


# ---------------------------------------------------------------- similarity
NGRAM_SIZE = 5                # word n-gram used for containment score
TFIDF_WEIGHT = 0.5            # weights must sum to 1.0
CONTAINMENT_WEIGHT = 0.5

# A chunk counts as copied at or above this combined score (0-1)
MATCH_THRESHOLD = 0.62
# Below this it is not even reported as a weak match
REPORT_THRESHOLD = 0.40

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "may",
    "more", "most", "not", "of", "on", "or", "our", "should", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "was", "we", "were", "what", "when", "which",
    "while", "who", "will", "with", "would", "you", "your",
}
