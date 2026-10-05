"""
app.py
------
Web interface for the plagiarism checker.

    python app.py         then open http://127.0.0.1:5000

Checks run in a background thread so the browser can poll a live progress
bar instead of staring at a frozen page for two minutes.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from pathlib import Path

from flask import (
    Flask, abort, jsonify, render_template, request, send_file, url_for
)
from markupsafe import Markup
from werkzeug.utils import secure_filename

import config
from detector import report as reporting
from detector.engine import PlagiarismEngine
from detector.extractor import ExtractionError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

# job_id -> {"status", "stage", "current", "total", "message", "report", "error"}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _update(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _run_check(job_id: str, source: dict, backend: str) -> None:
    def progress(stage, current, total, message):
        _update(job_id, stage=stage, current=current, total=total, message=message)

    try:
        engine = PlagiarismEngine(backend=backend, progress_callback=progress)
        if source["kind"] == "file":
            result = engine.check_file(source["path"])
        else:
            result = engine.check_text(source["text"], filename="Pasted text")
        _update(job_id, status="done", report=result, message="Finished")
    except ExtractionError as exc:
        _update(job_id, status="error", error=str(exc))
    except Exception as exc:                      # noqa: BLE001
        traceback.print_exc()
        _update(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------- routes
@app.get("/")
def index():
    return render_template(
        "index.html",
        backend=config.SEARCH_BACKEND,
        max_mb=config.MAX_UPLOAD_MB,
        extensions=", ".join(sorted(config.ALLOWED_EXTENSIONS)),
    )


@app.post("/check")
def check():
    backend = request.form.get("backend") or config.SEARCH_BACKEND
    pasted = (request.form.get("text") or "").strip()
    upload = request.files.get("document")

    if upload and upload.filename:
        filename = secure_filename(upload.filename)
        if Path(filename).suffix.lower() not in config.ALLOWED_EXTENSIONS:
            return jsonify(error=f"Only these file types work: "
                                 f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}"), 400
        path = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
        upload.save(path)
        source = {"kind": "file", "path": path}
    elif len(pasted.split()) >= config.CHUNK_MIN_WORDS:
        source = {"kind": "text", "text": pasted}
    else:
        return jsonify(
            error="Add a file, or paste at least "
                  f"{config.CHUNK_MIN_WORDS} words of text."
        ), 400

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running", "stage": "starting", "current": 0,
            "total": 1, "message": "Preparing the document", "report": None,
            "error": "",
        }
    threading.Thread(
        target=_run_check, args=(job_id, source, backend), daemon=True
    ).start()
    return jsonify(job_id=job_id)


@app.get("/progress/<job_id>")
def progress(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    return jsonify(
        status=job["status"],
        stage=job["stage"],
        current=job["current"],
        total=job["total"],
        message=job["message"],
        error=job["error"],
        result_url=url_for("result", job_id=job_id) if job["status"] == "done" else "",
    )


@app.get("/result/<job_id>")
def result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        abort(404)
    if job["status"] == "error":
        return render_template("result.html", error=job["error"], report=None), 500
    if job["status"] != "done":
        return render_template("result.html", pending=True, report=None)

    document = job["report"]
    return render_template(
        "result.html",
        report=document,
        body=Markup(reporting.render_body(document)),
        job_id=job_id,
    )


@app.get("/download/<job_id>.<fmt>")
def download(job_id: str, fmt: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    document = job["report"]
    stem = Path(document.filename).stem or "document"

    if fmt == "html":
        path = config.REPORT_DIR / f"{stem}_{job_id}.html"
        reporting.save_html(document, path)
    elif fmt == "json":
        path = config.REPORT_DIR / f"{stem}_{job_id}.json"
        reporting.save_json(document, path)
    else:
        abort(404)
    return send_file(path, as_attachment=True)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"That file is over {config.MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    print(f"Search backend: {config.SEARCH_BACKEND}")
    print("Open http://127.0.0.1:5000")
    app.run(debug=True, threaded=True)
