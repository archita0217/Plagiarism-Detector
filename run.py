"""
run.py
------
Command line interface.

    python run.py samples/copied_essay.txt
    python run.py essay.docx --backend google_cse --max-queries 15
    python run.py essay.txt --local samples/          (offline comparison)
    python run.py essay.txt --no-open --json
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import config
from detector import report as reporting
from detector.engine import PlagiarismEngine
from detector.extractor import ExtractionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Check a document against content published on the web.",
    )
    parser.add_argument("document", help="file to check (.txt .md .pdf .docx)")
    parser.add_argument(
        "--backend", default=config.SEARCH_BACKEND,
        choices=["duckduckgo", "google_cse", "serper"],
        help="search engine to use (default: %(default)s)",
    )
    parser.add_argument(
        "--local", metavar="DIR",
        help="compare against a folder of documents instead of the web",
    )
    parser.add_argument(
        "--max-queries", type=int, default=config.MAX_CHUNKS_TO_SEARCH,
        help="how many passages to search for (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold", type=float, default=config.MATCH_THRESHOLD,
        help="score at which a passage counts as copied (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="also save a JSON report")
    parser.add_argument("--no-html", action="store_true", help="skip the HTML report")
    parser.add_argument("--no-open", action="store_true",
                        help="do not open the report in a browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config.MAX_CHUNKS_TO_SEARCH = args.max_queries
    config.MATCH_THRESHOLD = args.threshold

    path = Path(args.document)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    def progress(stage, current, total, message):
        if total > 1:
            print(f"  [{stage}] {current}/{total} {message}")
        else:
            print(f"  [{stage}] {message}")

    engine = PlagiarismEngine(backend=args.backend, progress_callback=progress)
    print(f"Checking {path.name}")

    try:
        if args.local:
            print(f"Comparing against the documents in {args.local}")
            result = engine.check_against_local(path, args.local)
        else:
            result = engine.check_file(path)
    except ExtractionError as exc:
        print(f"Could not read the document: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"Could not run the check: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130

    reporting.print_console(result)

    if args.json:
        print(f" JSON report: {reporting.save_json(result)}")
    if not args.no_html:
        html_path = reporting.save_html(result)
        print(f" HTML report: {html_path}")
        if not args.no_open:
            webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
