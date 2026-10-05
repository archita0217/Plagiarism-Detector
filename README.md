# Web-Connected Plagiarism Detector

Detects plagiarism between an uploaded document and content published on the
internet, using text vectorisation and similarity measures.

The tool splits a document into passages, sends each passage to a search
engine as an exact phrase, downloads the pages that come back, and scores
every passage against them with TF-IDF cosine similarity and n-gram
containment. The output is a similarity percentage, a ranked list of matched
source URLs, and a marked-up copy of the document showing exactly which
sentences were found elsewhere and where.

---

## 1. Quick start

```bash
# 1. get into the project folder
cd plagiarism_detector

# 2. create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows:  venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4a. run the web interface
python app.py                     # then open http://127.0.0.1:5000

# 4b. or run it from the command line
python run.py samples/copied_essay.txt
```

No API key is needed. The default search backend is DuckDuckGo, which is free
and key-less.

To confirm everything works without an internet connection:

```bash
python test_offline.py
```

Expected output: the copied sample scores above 40%, the original sample
scores 0%, and a demo report is written to `reports/offline_demo.html`.

---

## 2. What the tool does

| Stage | Module | What happens |
|---|---|---|
| 1. Read | `detector/extractor.py` | `.pdf`, `.docx`, `.txt`, `.md`, `.rtf` → plain text, with PDF line-wrap repair |
| 2. Split | `detector/preprocess.py` | text → passages of ~35 words, never breaking a sentence |
| 3. Search | `detector/search.py` | each passage → an exact-phrase web query → candidate URLs |
| 4. Fetch | `detector/fetcher.py` | parallel download, boilerplate stripped, results cached on disk |
| 5. Compare | `detector/similarity.py` | TF-IDF cosine + 5-gram containment for every passage/source pair |
| 6. Decide | `detector/engine.py` | combine the scores, flag passages, compute the document percentage |
| 7. Report | `detector/report.py` | marked-up HTML, JSON, console summary |

```
                ┌──────────────┐
  upload  ───►  │  extractor   │  pdf / docx / txt  ──►  plain text
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │  preprocess  │  sentences ──► ~35-word passages
                └──────┬───────┘
                       ▼
        ┌──────────────────────────────┐
        │  search  (DuckDuckGo /       │   "exact phrase"  ──►  URLs
        │           Google CSE /       │
        │           Serper)            │
        └──────────────┬───────────────┘
                       ▼
                ┌──────────────┐
                │   fetcher    │  parallel GET, robots.txt, HTML → text, cache
                └──────┬───────┘
                       ▼
        ┌──────────────────────────────┐
        │  similarity                  │
        │   TF-IDF cosine   (0.5)      │  ──►  score per passage
        │   5-gram containment (0.5)   │
        │   longest common phrase      │  ──►  evidence to display
        └──────────────┬───────────────┘
                       ▼
                ┌──────────────┐
                │    report    │  % similarity, source list, marked document
                └──────────────┘
```

---

## 3. The similarity measures

Two measures are combined because each one alone is easy to fool.

### TF-IDF cosine similarity

Every passage becomes a vector. The weight of a term is

```
tfidf(t, d) = tf(t, d) × idf(t)
idf(t)      = log((1 + N) / (1 + df(t))) + 1
```

`tf` is how often the term appears in the passage, `df` is how many passages
in the whole corpus contain it, `N` is the number of passages. Rare words
therefore weigh far more than "the" or "is". Vectors are L2-normalised, so
similarity is the cosine of the angle between them:

```
cos(A, B) = (A · B) / (‖A‖ ‖B‖)          result in [0, 1]
```

The vectoriser is fitted on the document passages *and* the source passages
together, because `idf` must come from one shared corpus for the two sets of
vectors to be comparable. Unigrams and bigrams are both used, and sublinear
term frequency (`1 + log tf`) stops a repeated word from dominating.

Strength: survives reordering and light paraphrasing.
Weakness: two independently written texts on the same topic look similar.

### N-gram containment

With `G_A` the set of 5-word sequences in the passage and `G_B` those in the
source page:

```
containment(A, B) = |G_A ∩ G_B| / |G_A|
```

Matching many exact five-word sequences is very strong evidence of copying,
because word order has to match too.

Strength: almost no false positives on verbatim copying.
Weakness: heavy rewording defeats it.

### Combining them

```
score = 0.5 × cosine + 0.5 × containment
```

A passage is marked **copied** at `score ≥ 0.62` and **partly similar** at
`≥ 0.40`. The headline percentage is

```
similarity % = (words inside copied passages / total words) × 100
```

A third measure, the **longest common word sequence** (dynamic programming,
two-row longest-common-substring over words), is not used in the score. It
extracts the actual shared phrase so the report can show a human the evidence.

All thresholds and weights live in `config.py`.

---

## 4. Search backends

| Backend | Key needed | Free allowance | Setting |
|---|---|---|---|
| DuckDuckGo | no | unmetered, rate limited | `SEARCH_BACKEND=duckduckgo` |
| Google Programmable Search | yes | 100 queries/day | `SEARCH_BACKEND=google_cse` |
| Serper | yes | 2,500 queries on signup | `SEARCH_BACKEND=serper` |

To use a keyed backend, copy `.env.example` to `.env` and fill in the values:

```
SEARCH_BACKEND=google_cse
GOOGLE_API_KEY=your-key
GOOGLE_CSE_ID=your-search-engine-id
```

Get the Google key from the Cloud console and the search engine ID from
`programmablesearchengine.google.com` — set the engine to search the entire
web, not a specific site. If a keyed backend fails or runs out of quota, the
tool falls back to DuckDuckGo automatically, and stops searching altogether
after three consecutive failures so you are not left waiting.

---

## 5. Command line options

```bash
python run.py DOCUMENT [options]

  --backend {duckduckgo,google_cse,serper}   search engine to use
  --local DIR      compare against a folder of documents instead of the web
  --max-queries N  how many passages to search for (default 25)
  --threshold F    score at which a passage counts as copied (default 0.62)
  --json           also write a JSON report
  --no-html        skip the HTML report
  --no-open        do not open the report in a browser
```

Examples:

```bash
python run.py essay.docx
python run.py essay.pdf --backend serper --max-queries 15
python run.py essay.txt --local samples/ --json
```

`--local` is the offline mode: it compares one file against every other
document in a folder. Useful for detecting copying between classmates, and for
demonstrating the tool when there is no internet in the exam hall.

---

## 6. Project layout

```
plagiarism_detector/
├── app.py               Flask web interface (background jobs + progress bar)
├── run.py               command line interface
├── config.py            every tunable setting and threshold
├── test_offline.py      self-test that needs no internet
├── requirements.txt
├── .env.example         template for API keys
├── detector/
│   ├── extractor.py     file  → text
│   ├── preprocess.py    text  → sentences → passages, n-grams, tokens
│   ├── search.py        passage → candidate URLs (3 backends)
│   ├── fetcher.py       URL → page text (threaded, cached, robots-aware)
│   ├── similarity.py    TF-IDF, cosine, containment, Jaccard, LCS
│   ├── engine.py        the pipeline; produces a Report object
│   └── report.py        Report → HTML / JSON / console
├── templates/           base.html, index.html, result.html
├── static/style.css
├── samples/             three documents for testing
├── uploads/  cache/  reports/       created automatically
```

---

## 7. Engineering details worth mentioning in a viva

**Why passages instead of whole documents.** Comparing two whole documents
gives one blurry number. Comparing ~35-word passages tells you *which*
sentences were copied and *from where*, which is what a plagiarism report
actually needs. Passages never break a sentence, so each one also makes a
sensible search query.

**Why passages are closed early.** A passage that overshoots its target mixes
copied sentences with original ones and drags a real match below the
threshold, so a passage is closed before it can exceed about 1.3× the target
size.

**Which passages get searched.** Searching every passage of a 5,000-word
report would mean hundreds of queries. `select_query_chunks` picks an evenly
spread subset (default 25) and skips passages with fewer than eight
content words, because generic sentences return useless results.

**Quoted queries.** Queries are sent wrapped in quotation marks, so verbatim
copying jumps straight to the top of the results. If the quoted search returns
nothing — normal for original writing — the query is retried unquoted so
paraphrased sources can still surface.

**Politeness and reliability.** A delay between searches, a descriptive
User-Agent, `robots.txt` checking, a 72-hour disk cache, and a six-thread
download pool. Re-running a check on the same document costs almost nothing.

**Graceful degradation.** Missing scikit-learn falls back to a pure-Python
TF-IDF implementation of the same formulae. Missing BeautifulSoup falls back
to regex HTML stripping. A failing search backend falls back to DuckDuckGo.
The tool reports what went wrong rather than crashing.

---

## 8. Known limits

- **Recall is bounded by the search engine.** Only pages that a search engine
  can find and that allow crawling are compared. Paywalled journals, PDFs
  behind logins and printed books are invisible to it.
- **Heavy paraphrasing is missed.** Cosine similarity on TF-IDF vectors is a
  lexical measure. Rewritten sentences with synonyms score low. Detecting that
  properly needs sentence embeddings (for example `sentence-transformers`),
  which is the obvious next version of this project.
- **Quotations and references inflate the score.** Correctly quoted and cited
  material matches its source, exactly as copied material does. The percentage
  is evidence for a human to interpret, not a verdict.
- **Common phrases can match.** Boilerplate such as a standard definition may
  be flagged; the containment half of the score keeps this rare but not
  impossible.
- **Rate limits.** DuckDuckGo may throttle rapid repeated use. Lower
  `MAX_CHUNKS_TO_SEARCH` or raise `SEARCH_DELAY_SECONDS` in `config.py`.

---

## 9. Possible extensions

- Semantic matching with sentence embeddings, combined with the current
  lexical score.
- The supervised route from the problem statement: label pairs of passages as
  copied or not, build features (cosine, containment, Jaccard, longest common
  phrase length), and train a logistic regression or random forest to replace
  the hand-set threshold.
- A database of previously checked submissions, so a batch of assignments can
  be checked against each other as well as against the web.
- Citation-aware scoring that ignores text inside quotation marks with a
  reference attached.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `No web results came back` | Check the connection; try `--backend serper`; DuckDuckGo may be rate limiting you |
| `DuckDuckGo backend needs a package` | `pip install ddgs` |
| Google returns quota errors | The 100 free daily queries are used up; wait or switch backend |
| PDF gives "too little readable text" | It is a scanned image; run OCR first (for example `ocrmypdf in.pdf out.pdf`) |
| Everything scores 0% | Run `python test_offline.py` — if that passes, the maths is fine and the problem is the search step |
| The check is slow | Lower `MAX_CHUNKS_TO_SEARCH` and `MAX_SOURCES_TO_FETCH` in `config.py` |
