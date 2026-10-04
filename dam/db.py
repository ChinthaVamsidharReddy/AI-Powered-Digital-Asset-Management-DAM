"""SQLite persistence (WAL). One connection per thread; explicit short write transactions."""
from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (          -- one row per unique CONTENT (sha256)
    id          INTEGER PRIMARY KEY,
    sha256      TEXT UNIQUE NOT NULL,
    type        TEXT NOT NULL,               -- image | video | pdf
    ext         TEXT,
    mime        TEXT,
    size        INTEGER NOT NULL,
    width       INTEGER, height INTEGER,
    duration    REAL,                        -- seconds (video)
    pages       INTEGER,                     -- pdf
    meta        TEXT,                        -- json: exif, codec, pdf info, ...
    status      TEXT NOT NULL,               -- ready | failed
    error       TEXT,
    warnings    TEXT,                        -- json list (partial problems)
    thumb       TEXT,
    pipeline    TEXT NOT NULL,               -- models/config used; mismatch => re-index
    created_ts  REAL, indexed_ts REAL
);
CREATE TABLE IF NOT EXISTS files (           -- one row per PATH on disk (also the job queue)
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    folder      TEXT, name TEXT, ext TEXT,
    size        INTEGER, mtime_ns INTEGER,
    asset_id    INTEGER REFERENCES assets(id) ON DELETE SET NULL,
    state       TEXT NOT NULL,               -- pending | processing | done | failed | unsupported
    error       TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    present     INTEGER NOT NULL DEFAULT 1,
    seen_run    INTEGER,
    updated_ts  REAL
);
CREATE INDEX IF NOT EXISTS files_state ON files(state, present);
CREATE INDEX IF NOT EXISTS files_asset ON files(asset_id);
CREATE TABLE IF NOT EXISTS vectors (         -- embeddings: images, video frames, pdf pages, text chunks
    id        INTEGER PRIMARY KEY,
    asset_id  INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    space     TEXT NOT NULL,                 -- clip | text
    kind      TEXT NOT NULL,                 -- image | frame | page | chunk | meta | transcript
    ref       REAL,                          -- seconds (frame/transcript) or 1-based page
    text      TEXT,
    thumb     TEXT,
    vec       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS vectors_asset ON vectors(asset_id);
CREATE INDEX IF NOT EXISTS vectors_space ON vectors(space);
CREATE VIRTUAL TABLE IF NOT EXISTS vtext USING fts5(text, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, started_ts REAL, finished_ts REAL,
    status TEXT, summary TEXT
);
"""


class Database:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.conn().executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=60, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=60000")
            self._local.c = c
        return c

    @contextlib.contextmanager
    def tx(self):
        c = self.conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
