
from __future__ import annotations
import html, json, math
from pathlib import Path
import config
from detector.engine import Report
_CSS_FILE = config.BASE_DIR / "static" / "style.css"

def _esc(text: str) -> str:
    return html.escape(text or "", quote=True)

def _tone(pct: float) -> str:
    if pct >= 40: return "high"
    if pct >= 15: return "mid"
    if pct >= 5: return "low"
    return "clean"

def _score_ring(report: Report) -> str:
    pct = float(report.similarity_percent)
    c = 2 * math.pi * 70
    covered = c * min(max(pct, 0), 100) / 100
    label = f"{pct:g}" if pct == int(pct) else f"{pct:.1f}"
    return f'''
<div class="score-ring" role="img" aria-label="{label} percent">
<svg viewBox="0 0 160 160"><circle class="track" cx="80" cy="80" r="70"></circle>
<circle class="fill" cx="80" cy="80" r="70" stroke-dasharray="{covered:.1f} {c:.1f}"></circle></svg>
<div class="score-ring-content"><strong>{label}<sup>%</sup></strong></div></div>'''

def _strip(report: Report) -> str:
    if not report.matches: return ""
    total = sum(m.word_count for m in report.matches) or 1
    bands = "".join(f'<span class="s-{m.verdict}" style="width:{100*m.word_count/total:.3f}%"></span>' for m in report.matches)
    return ('<div class="match-strip">'+bands+'</div>'
            '<div class="strip-key"><span><i class="key-copied"></i>Matched</span>'
            '<span><i class="key-similar"></i>Similar</span><span><i class="key-original"></i>Unflagged</span></div>')

def _stats(report: Report) -> str:
    cells=[("Passages checked",report.total_chunks),("Web searches",report.queries_made),
           ("Pages compared",report.pages_fetched),("Sources matched",len(report.sources)),
           ("Time taken",f"{report.duration_seconds:g}s")]
    return '<div class="stat-grid">'+''.join(
        f'<div class="stat"><b>{_esc(str(v))}</b><span>{_esc(str(k))}</span></div>' for k,v in cells)+'</div>'

def _verdict(report: Report) -> str:
    where = "your comparison corpus" if report.backend == "local corpus" else "web sources"
    return f'''
<div class="report-summary tone-{_tone(float(report.similarity_percent))}">{_score_ring(report)}
<div class="summary-copy"><span class="verdict-chip">{_esc(report.verdict)}</span>
<h2>Plagiarism overview</h2>
<p>{report.copied_words} of {report.total_words} words fall inside passages that matched {where}.</p>
{_strip(report)}</div></div>{_stats(report)}'''

def _sources_cards(report: Report) -> str:
    if not report.sources: return '<p class="empty">No source matched strongly enough to be listed.</p>'
    rows=[]
    for i, source in enumerate(report.sources,1):
        rows.append(f'''
<article class="source-item"><span class="source-num">{i:02d}</span>
<div class="source-main"><a class="source-title" href="{_esc(source.url)}" target="_blank" rel="noopener">{_esc(source.title[:140] or source.url)}</a>
<div class="source-domain">{_esc(source.domain)}</div></div>
<div class="source-score"><b>{source.percent_of_document:g}%</b><span>{source.words_matched} matched words</span></div></article>''')
    return '<div class="source-list">'+''.join(rows)+'</div>'

def _marked_document(report: Report) -> str:
    out=[]
    for m in report.matches:
        text=_esc(m.chunk_text)
        if m.verdict=="original":
            out.append(f'<p><span class="passage">{text}</span></p>'); continue
        note="note" if m.verdict=="copied" else "note similar-note"
        excerpt=f' Longest identical phrase: "{_esc(m.matched_excerpt[:160])}".' if m.matched_excerpt else ""
        out.append(f'<p><span class="passage {m.verdict}">{text}</span><span class="{note}">Matches <a href="{_esc(m.source_url)}" target="_blank" rel="noopener">{_esc(m.source_domain or m.source_url)}</a> <span class="figures">{round(m.score*100)}% match</span>.{excerpt}</span></p>')
    return ''.join(out)

def _warnings(report: Report) -> str:
    if not report.warnings: return ""
    return '<div class="warning"><strong>Review notes</strong>'+''.join(f'<div>{_esc(w)}</div>' for w in report.warnings)+'</div>'

def render_body(report: Report) -> str:
    return f'''
{_warnings(report)}
<section class="report-card summary-card">{_verdict(report)}</section>
<section class="report-card"><header><h2>Matched sources</h2><p>Open a source to verify the context behind each finding.</p></header>{_sources_cards(report)}</section>
<section class="report-card"><header><h2>Document evidence</h2><p>Highlighted passages are connected to their strongest matching source.</p></header>
<div class="legend"><span><i class="l-copied"></i>Matched a source</span><span><i class="l-similar"></i>Partly similar</span></div>
<div class="document-wrap"><div class="document">{_marked_document(report)}</div></div></section>'''

def build_standalone_html(report: Report) -> str:
    try: css=_CSS_FILE.read_text(encoding="utf-8")
    except Exception: css=""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PlagiScan plagiarism report · {_esc(report.filename)}</title>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet"><style>{css}</style></head>
<body><main class="app-shell report-page"><div class="report-top"><div><div class="report-label">Plagiarism report</div>
<h1>{_esc(report.filename)}</h1><div class="report-meta">Checked {_esc(report.checked_at)}</div></div></div>{render_body(report)}</main></body></html>'''

def save_html(report: Report, path: str | Path | None = None) -> Path:
    path=Path(path) if path else config.REPORT_DIR/f"{Path(report.filename).stem}_report.html"
    path.write_text(build_standalone_html(report),encoding="utf-8"); return path

def save_json(report: Report, path: str | Path | None = None) -> Path:
    path=Path(path) if path else config.REPORT_DIR/f"{Path(report.filename).stem}_report.json"
    path.write_text(json.dumps(report.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8"); return path

def print_console(report: Report) -> None:
    line="-"*62
    print(f"\\n{line}\\n PLAGISCAN PLAGIARISM REPORT · {report.filename}\\n{line}")
    print(f" Plagiarism : {report.similarity_percent:g}% ({report.verdict})")
    print(f" Words matched : {report.copied_words} of {report.total_words}")
    print(f" Passages checked : {report.total_chunks}")
    print(f" Web searches : {report.queries_made}")
    print(f" Pages compared : {report.pages_fetched}")
    print(f" Time taken : {report.duration_seconds:g}s")
