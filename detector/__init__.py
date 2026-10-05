"""
detector
--------
A web-connected plagiarism checker built on classic information-retrieval
measures: TF-IDF vectorisation, cosine similarity and n-gram containment.

Typical use:

    from detector.engine import PlagiarismEngine
    from detector import report

    result = PlagiarismEngine().check_file("essay.docx")
    report.print_console(result)
    report.save_html(result)
"""

__version__ = "1.0.0"
__all__ = [
    "engine",
    "extractor",
    "fetcher",
    "preprocess",
    "report",
    "search",
    "similarity",
]
