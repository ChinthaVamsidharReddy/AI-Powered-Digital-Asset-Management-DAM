from __future__ import annotations

import mimetypes
import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings
from .db import Database
from .embedders import make_embedder
from .indexer import Indexer
from .search import Searcher

STATIC = Path(__file__).parent / "static"


class IndexRequest(BaseModel):
    retry_failed: bool = False
    force: bool = False


def _csv(v):
    return [x.strip().lower() for x in v.split(",") if x.strip()] if v else None


def create_app(settings: Settings | None = None, embedder=None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings.db_path)
    embedder = embedder or make_embedder(settings)
    indexer = Indexer(settings, db, embedder)
    indexer.recover()
    searcher = Searcher(settings, db, embedder)
    settings.thumbs_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="AI Digital Asset Management", version="1.0.0")
    app.state.indexer, app.state.searcher, app.state.db = indexer, searcher, db
    threading.Thread(target=lambda: _safe(embedder.warmup), daemon=True).start()  # preload models in background

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    app.mount("/thumbs", StaticFiles(directory=str(settings.thumbs_dir)), name="thumbs")

    # ------------------------------------------------------------------ search
    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1), types: str | None = None, ext: str | None = None,
               min_size_mb: float | None = None, max_size_mb: float | None = None, folder: str | None = None,
               indexed_after: float | None = None, indexed_before: float | None = None,
               limit: int = Query(24, ge=1, le=100), offset: int = Query(0, ge=0)):
        try:
            return searcher.search(
                q, types=_csv(types), exts=_csv(ext),
                min_size=min_size_mb * 1024 * 1024 if min_size_mb is not None else None,
                max_size=max_size_mb * 1024 * 1024 if max_size_mb is not None else None,
                folder=folder or None, after=indexed_after, before=indexed_before, limit=limit, offset=offset)
        except ValueError as e:
            raise HTTPException(400, str(e))

    # ------------------------------------------------------------------ indexing
    @app.post("/api/index/start")
    def index_start(req: IndexRequest = IndexRequest()):
        try:
            indexer.start(retry_failed=req.retry_failed, force=req.force)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return indexer.status()

    @app.post("/api/index/cancel")
    def index_cancel():
        indexer.cancel()
        return {"cancelling": True}

    @app.get("/api/index/status")
    def index_status():
        return indexer.status()

    # ------------------------------------------------------------------ catalogue
    @app.get("/api/stats")
    def stats():
        c = db.conn()
        by_type = [dict(r) for r in c.execute(
            "SELECT a.type, COUNT(DISTINCT a.id) assets, COUNT(f.id) files, SUM(a.size) bytes FROM assets a "
            "JOIN files f ON f.asset_id=a.id AND f.present=1 WHERE a.status='ready' GROUP BY a.type")]
        by_state = {r["state"]: r["n"] for r in c.execute("SELECT state, COUNT(*) n FROM files WHERE present=1 GROUP BY state")}
        dup = c.execute("SELECT COUNT(*) FROM (SELECT asset_id FROM files WHERE present=1 AND asset_id IS NOT NULL "
                        "GROUP BY asset_id HAVING COUNT(*)>1)").fetchone()[0]
        total_bytes = c.execute("SELECT COALESCE(SUM(size),0) FROM files WHERE present=1").fetchone()[0]
        vec = {r["space"]: r["n"] for r in c.execute("SELECT space, COUNT(*) n FROM vectors GROUP BY space")}
        return {"by_type": by_type, "files_by_state": by_state, "duplicate_groups": dup,
                "total_bytes_on_disk": total_bytes, "vectors": vec}

    @app.get("/api/files")
    def list_files(state: str = "failed,unsupported", limit: int = Query(200, ge=1, le=1000), offset: int = 0):
        states = _csv(state) or ["failed"]
        rows = db.conn().execute(
            f"SELECT id,path,name,ext,size,state,error,attempts FROM files WHERE present=1 AND state IN ({','.join('?' * len(states))}) "
            f"ORDER BY state, path LIMIT ? OFFSET ?", (*states, limit, offset)).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/duplicates")
    def duplicates():
        c = db.conn()
        groups = []
        for r in c.execute("SELECT asset_id FROM files WHERE present=1 AND asset_id IS NOT NULL GROUP BY asset_id "
                           "HAVING COUNT(*)>1 ORDER BY COUNT(*) DESC LIMIT 200"):
            paths = [x[0] for x in c.execute("SELECT path FROM files WHERE asset_id=? AND present=1 ORDER BY path", (r[0],))]
            groups.append({"asset_id": r[0], "paths": paths})
        return groups

    # ------------------------------------------------------------------ file access
    def _file(file_id: int):
        row = db.conn().execute("SELECT * FROM files WHERE id=? AND present=1", (file_id,)).fetchone()
        if not row or not os.path.isfile(row["path"]):
            raise HTTPException(404, "file not found on disk")
        return row

    @app.get("/api/files/{file_id}/content")
    def file_content(file_id: int):
        row = _file(file_id)
        mt = mimetypes.guess_type(row["path"])[0] or "application/octet-stream"
        return FileResponse(row["path"], media_type=mt, content_disposition_type="inline")

    @app.post("/api/files/{file_id}/reveal")
    def reveal(file_id: int):
        """Open the OS file manager at the original file (local use only)."""
        row = _file(file_id)
        p = os.path.abspath(row["path"])
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", f"/select,{p}"])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", p])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(p)])
        except OSError as e:
            raise HTTPException(500, f"could not open file manager: {e}")
        return {"opened": p}

    return app


def _safe(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001  (model download problems surface on first use too)
        import logging
        logging.getLogger("dam").exception("model warmup failed")
