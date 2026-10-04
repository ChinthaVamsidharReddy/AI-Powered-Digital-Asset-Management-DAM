"""Incremental, crash-safe indexer.

Phases of a run:  scan (cheap stat() walk, upserts `files` rows)  ->  process (worker threads)  ->  gc.

State machine for a file row:  pending -> processing -> done | failed | unsupported
* The `files` table IS the job queue, so progress survives restarts. Rows left in 'processing' by a
  crash are reset to 'pending' on startup (`recover`).
* A file is skipped when (size, mtime_ns) are unchanged and it is done/unsupported/failed.
* Content-hash (sha256) de-duplication: identical bytes in different folders share ONE asset
  (one set of embeddings); every path is still recorded and shown in the UI.
* Each file is committed in a single transaction, so a crash never leaves half-indexed assets.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import stat as statmod
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .embedders import EmbedderError
from .extractors import CorruptFile, UnsupportedFile, extract, type_for_ext

log = logging.getLogger("dam.indexer")
IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class RunStats:
    phase: str = "idle"          # idle | scanning | processing | done | cancelled | error
    started: float = 0.0
    finished: float = 0.0
    scanned: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    queued: int = 0
    processed: int = 0
    indexed: int = 0             # fully analysed by AI this run
    reused: int = 0              # content already indexed (duplicate / touched file) - no AI work
    failed: int = 0
    unsupported: int = 0
    error: str | None = None


class Indexer:
    def __init__(self, settings, db, embedder):
        self.s, self.db, self.embedder = settings, db, embedder
        self.pipeline = embedder.pipeline
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.stats = RunStats()
        self._sha_locks = [threading.Lock() for _ in range(1024)]
        self._current: dict[int, str] = {}
        self._cur_lock = threading.Lock()
        self._vector_listeners = []
        s = self.s
        s.thumbs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def recover(self):
        """Crash recovery: anything left mid-flight goes back to the queue."""
        self.db.conn().execute("UPDATE files SET state='pending' WHERE state='processing'")

    def start(self, retry_failed=False, force=False, background=True):
        with self._start_lock:
            if self.running:
                raise RuntimeError("indexing is already running")
            self._cancel.clear()
            self.stats = RunStats(phase="scanning", started=time.time())
            if background:
                self._thread = threading.Thread(target=self._run, args=(retry_failed, force), daemon=True, name="indexer")
                self._thread.start()
                return
        self._run(retry_failed, force)

    def cancel(self):
        self._cancel.set()

    def wait(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ run
    def _run(self, retry_failed, force):
        st = self.stats
        c = self.db.conn()
        run_id = c.execute("INSERT INTO runs(started_ts,status) VALUES(?,?)", (st.started, "running")).lastrowid
        try:
            self.recover()
            c.execute("UPDATE files SET state='pending', attempts=0 WHERE present=1 AND state='done' AND asset_id IN "
                      "(SELECT id FROM assets WHERE pipeline != ?)", (self.pipeline,))
            complete = self._scan(run_id, retry_failed, force)
            if complete:
                self._mark_missing(run_id)
            st.phase = "processing"
            self._process_pending()
            if not self._cancel.is_set():
                self._gc_orphans()
            st.phase = "cancelled" if self._cancel.is_set() else "done"
        except Exception as e:  # noqa: BLE001
            log.exception("indexing run failed")
            st.phase, st.error = "error", f"{type(e).__name__}: {e}"
        finally:
            st.finished = time.time()
            c.execute("UPDATE runs SET finished_ts=?, status=?, summary=? WHERE id=?",
                      (st.finished, st.phase, json.dumps(asdict(st)), run_id))

    # ------------------------------------------------------------------ scan
    def _scan(self, run_id, retry_failed, force) -> bool:
        root = self.s.media_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"MEDIA_DIR does not exist: {root}")
        batch = []
        for dirpath, dirnames, filenames in os.walk(root):
            if self._cancel.is_set():
                return False
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for fn in sorted(filenames):
                if fn.startswith(".") or fn.lower() in IGNORED_NAMES:
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    stt = os.stat(full)
                except OSError:
                    continue
                if not statmod.S_ISREG(stt.st_mode):
                    continue
                batch.append((full, dirpath, fn, os.path.splitext(fn)[1].lower(), stt.st_size, stt.st_mtime_ns))
                if len(batch) >= 500:
                    self._ingest_batch(batch, run_id, retry_failed, force)
                    batch = []
        if batch:
            self._ingest_batch(batch, run_id, retry_failed, force)
        return True

    def _ingest_batch(self, batch, run_id, retry_failed, force):
        st, now = self.stats, time.time()
        with self.db.tx() as c:
            for full, folder, fn, ext, size, mtime in batch:
                st.scanned += 1
                row = c.execute("SELECT id,size,mtime_ns,state FROM files WHERE path=?", (full,)).fetchone()
                supported = type_for_ext(ext) is not None
                if row is None:
                    st.new += 1
                    state, err = ("pending", None) if supported else ("unsupported", f"unsupported file type '{ext or '(none)'}'")
                    if not supported:
                        st.unsupported += 1
                    c.execute("INSERT INTO files(path,folder,name,ext,size,mtime_ns,state,error,present,seen_run,updated_ts) "
                              "VALUES(?,?,?,?,?,?,?,?,1,?,?)", (full, folder, fn, ext, size, mtime, state, err, run_id, now))
                    continue
                same = row["size"] == size and row["mtime_ns"] == mtime
                redo = supported and (force or not same or row["state"] in ("pending", "processing", "unsupported")
                                      or (row["state"] == "failed" and retry_failed))
                if redo:
                    if not same:
                        st.changed += 1
                    c.execute("UPDATE files SET state='pending', error=NULL, attempts=0, size=?, mtime_ns=?, present=1, "
                              "seen_run=?, updated_ts=? WHERE id=?", (size, mtime, run_id, now, row["id"]))
                elif not supported:
                    st.unsupported += 1
                    c.execute("UPDATE files SET present=1, seen_run=?, state='unsupported', error=?, size=?, mtime_ns=? WHERE id=?",
                              (run_id, f"unsupported file type '{ext or '(none)'}'", size, mtime, row["id"]))
                else:
                    st.unchanged += 1
                    c.execute("UPDATE files SET present=1, seen_run=? WHERE id=?", (run_id, row["id"]))

    def _mark_missing(self, run_id):
        c = self.db.conn()
        n_seen = c.execute("SELECT COUNT(*) FROM files WHERE seen_run=?", (run_id,)).fetchone()[0]
        n_all = c.execute("SELECT COUNT(*) FROM files WHERE present=1").fetchone()[0]
        if n_seen == 0 and n_all > 0:
            log.warning("scan found no files but index has %d - drive unmounted? not marking anything missing", n_all)
            return
        c.execute("UPDATE files SET present=0 WHERE present=1 AND (seen_run IS NULL OR seen_run != ?)", (run_id,))

    # ------------------------------------------------------------------ process
    def _process_pending(self):
        c = self.db.conn()
        rows = c.execute("SELECT id, ext, size FROM files WHERE present=1 AND state='pending'").fetchall()
        order = {"image": 0, "pdf": 1, "video": 2}
        rows = sorted(rows, key=lambda r: (order.get(type_for_ext(r["ext"]), 3), r["size"]))  # quick wins first
        self.stats.queued = len(rows)
        q: queue.Queue = queue.Queue()
        for r in rows:
            q.put(r["id"])

        def worker():
            while not self._cancel.is_set():
                try:
                    fid = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    self._process_file(fid)
                except Exception:  # noqa: BLE001
                    log.exception("unhandled error processing file id=%s", fid)
                self.stats.processed += 1

        threads = [threading.Thread(target=worker, daemon=True, name=f"idx-worker-{i}") for i in range(max(1, self.s.workers))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _mark(self, fid, state, error=None):
        self.db.conn().execute("UPDATE files SET state=?, error=?, updated_ts=? WHERE id=?", (state, error, time.time(), fid))

    def _process_file(self, fid):
        c = self.db.conn()
        row = c.execute("SELECT * FROM files WHERE id=?", (fid,)).fetchone()
        if row is None or row["state"] != "pending":
            return
        c.execute("UPDATE files SET state='processing', attempts=attempts+1, updated_ts=? WHERE id=?", (time.time(), fid))
        path, ext, st = row["path"], row["ext"], self.stats
        with self._cur_lock:
            self._current[fid] = path
        try:
            try:
                stt = os.stat(path)
            except FileNotFoundError:
                c.execute("UPDATE files SET present=0, state='failed', error='file disappeared during indexing' WHERE id=?", (fid,))
                st.failed += 1
                return
            if stt.st_size == 0:
                raise CorruptFile("empty file (0 bytes)")
            sha = sha256_file(path)
            with self._sha_locks[int(sha[:3], 16) % 1024]:
                a = c.execute("SELECT id,status,pipeline FROM assets WHERE sha256=?", (sha,)).fetchone()
                if a and a["status"] == "ready" and a["pipeline"] == self.pipeline:
                    c.execute("UPDATE files SET asset_id=?, state='done', error=NULL, size=?, mtime_ns=?, updated_ts=? WHERE id=?",
                              (a["id"], stt.st_size, stt.st_mtime_ns, time.time(), fid))
                    st.reused += 1
                    return
                ex = extract(path, ext, self.s)
                self._store(fid, sha, stt, ex, ext)
            st.indexed += 1
        except UnsupportedFile as e:
            self._mark(fid, "unsupported", str(e))
            st.unsupported += 1
        except CorruptFile as e:
            self._mark(fid, "failed", f"corrupt or unreadable: {e}")
            st.failed += 1
        except EmbedderError as e:
            self._mark(fid, "failed", f"AI processing failed: {e}")
            st.failed += 1
        except Exception as e:  # noqa: BLE001
            log.exception("unexpected failure for %s", path)
            self._mark(fid, "failed", f"unexpected error: {type(e).__name__}: {e}")
            st.failed += 1
        finally:
            with self._cur_lock:
                self._current.pop(fid, None)

    # ------------------------------------------------------------------ AI + persistence
    def _retry(self, fn, tries=3):
        for i in range(tries):
            try:
                return fn()
            except EmbedderError:
                if i == tries - 1:
                    raise
                time.sleep(1.5 * (i + 1))

    def _save_thumb(self, img: Image.Image, rel: str):
        dest = self.s.thumbs_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        t = img.copy()
        t.thumbnail((self.s.thumb_size, self.s.thumb_size))
        tmp = dest.with_suffix(".tmp")
        t.convert("RGB").save(tmp, format="JPEG", quality=78)
        os.replace(tmp, dest)

    def _store(self, fid, sha, stt, ex, ext):
        s, now = self.s, time.time()
        base = dict(sha=sha, type=ex.type, ext=ext,
                    mime=ex.mime, size=stt.st_size, w=ex.width, h=ex.height, dur=ex.duration, pages=ex.pages,
                    meta=json.dumps(ex.meta, default=str), warn=json.dumps(ex.warnings))
        try:
            clip_rows, text_rows = self._embed(sha, ex)
        except EmbedderError as e:   # keep the metadata so the asset is visible in stats/failures, then fail the file
            with self.db.tx() as c:
                self._upsert_asset(c, base, "failed", str(e), None, now)
            raise
        if not clip_rows and not text_rows:
            raise CorruptFile("no content could be extracted")
        thumb = next((r[2] for r in clip_rows if r[2]), None)
        with self.db.tx() as c:
            aid = self._upsert_asset(c, base, "ready", None, thumb, now)
            old = [r[0] for r in c.execute("SELECT id FROM vectors WHERE asset_id=?", (aid,))]
            for i in range(0, len(old), 500):
                part = old[i:i + 500]
                c.execute(f"DELETE FROM vtext WHERE rowid IN ({','.join('?' * len(part))})", part)
            c.execute("DELETE FROM vectors WHERE asset_id=?", (aid,))
            for kind, ref, th, vec in clip_rows:
                c.execute("INSERT INTO vectors(asset_id,space,kind,ref,thumb,vec) VALUES(?,?,?,?,?,?)",
                          (aid, "clip", kind, ref, th, vec.astype("<f4").tobytes()))
            for kind, ref, text, vec in text_rows:
                vid = c.execute("INSERT INTO vectors(asset_id,space,kind,ref,text,vec) VALUES(?,?,?,?,?,?)",
                                (aid, "text", kind, ref, text, vec.astype("<f4").tobytes())).lastrowid
                c.execute("INSERT INTO vtext(rowid,text) VALUES(?,?)", (vid, text))
            c.execute("UPDATE files SET asset_id=?, state='done', error=NULL, size=?, mtime_ns=?, updated_ts=? WHERE id=?",
                      (aid, stt.st_size, stt.st_mtime_ns, now, fid))

    def _upsert_asset(self, c, b, status, error, thumb, now):
        c.execute(
            "INSERT INTO assets(sha256,type,ext,mime,size,width,height,duration,pages,meta,status,error,warnings,thumb,pipeline,created_ts,indexed_ts) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(sha256) DO UPDATE SET type=excluded.type, ext=excluded.ext, "
            "mime=excluded.mime, size=excluded.size, width=excluded.width, height=excluded.height, duration=excluded.duration, "
            "pages=excluded.pages, meta=excluded.meta, status=excluded.status, error=excluded.error, warnings=excluded.warnings, "
            "thumb=excluded.thumb, pipeline=excluded.pipeline, indexed_ts=excluded.indexed_ts",
            (b["sha"], b["type"], b["ext"], b["mime"], b["size"], b["w"], b["h"], b["dur"], b["pages"], b["meta"], status,
             error, b["warn"], thumb, self.pipeline, now, now))
        return c.execute("SELECT id FROM assets WHERE sha256=?", (b["sha"],)).fetchone()[0]

    def _embed(self, sha, ex):
        clip_rows, text_rows = [], []
        if ex.visuals:
            vecs = self._retry(lambda: self.embedder.encode_images([v.image for v in ex.visuals]))
            keep, last = [], None
            for i, v in enumerate(vecs):                       # drop near-identical consecutive frames
                if ex.type == "video" and last is not None and float(np.dot(v, last)) >= self.s.video_dedupe_threshold:
                    continue
                keep.append(i)
                last = v
            for n, i in enumerate(keep):
                vi = ex.visuals[i]
                rel = f"{sha[:2]}/{sha}_{n}.jpg"
                self._save_thumb(vi.image, rel)
                clip_rows.append((vi.kind, vi.ref, rel, vecs[i]))
        if ex.texts:
            tv = self._retry(lambda: self.embedder.encode_text([t.text for t in ex.texts]))
            for t, v in zip(ex.texts, tv):
                text_rows.append((t.kind, t.ref, t.text, v))
        return clip_rows, text_rows

    # ------------------------------------------------------------------ housekeeping / status
    def _gc_orphans(self):
        """Delete assets that no `files` row references any more (content replaced / file removed for good)."""
        c = self.db.conn()
        ids = [r[0] for r in c.execute("SELECT a.id FROM assets a WHERE NOT EXISTS (SELECT 1 FROM files f WHERE f.asset_id=a.id)")]
        for aid in ids:
            thumbs = [r[0] for r in c.execute("SELECT thumb FROM vectors WHERE asset_id=? AND thumb IS NOT NULL", (aid,))]
            with self.db.tx() as t:
                t.execute("DELETE FROM vtext WHERE rowid IN (SELECT id FROM vectors WHERE asset_id=?)", (aid,))
                t.execute("DELETE FROM assets WHERE id=?", (aid,))
            for th in thumbs:
                try:
                    os.remove(self.s.thumbs_dir / th)
                except OSError:
                    pass

    def status(self) -> dict:
        st, c = self.stats, self.db.conn()
        counts = {r["state"]: r["n"] for r in c.execute("SELECT state, COUNT(*) n FROM files WHERE present=1 GROUP BY state")}
        elapsed = (st.finished or time.time()) - st.started if st.started else 0
        rate = st.processed / elapsed if elapsed > 1 and st.processed else 0
        remaining = max(st.queued - st.processed, 0)
        with self._cur_lock:
            current = list(self._current.values())
        return {"running": self.running, "run": asdict(st), "counts": counts, "elapsed_sec": round(elapsed, 1),
                "rate_per_sec": round(rate, 2), "eta_sec": round(remaining / rate) if rate and self.running else None,
                "current": current, "pipeline": self.pipeline}
