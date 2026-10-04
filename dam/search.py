"""Hybrid semantic search.

Three signals per asset:
  clip : query -> CLIP text tower  vs. image / video-frame / PDF-page embeddings
  text : query -> MiniLM           vs. PDF-chunk / transcript embeddings
  kw   : query terms -> FTS5 BM25 over the same extracted text

Relevance control (important for result quality):
  * absolute floor per signal  - nothing below it is ever returned (negative queries -> ~0 results)
  * relative margin per signal - only hits within `margin` of the BEST hit for this query survive,
    so the tail of weakly related assets is cut adaptively instead of returning "everything"
  * ranking = weighted RRF per signal, combined as  max + 0.25 * (sum of the others)
    (plain sum let PDFs, which match on 3 signals, outrank images/videos that can only match on 1)
  * `confidence` = ABSOLUTE match strength (0..1) from the raw similarity, NOT relative to the top hit,
    so a poor best match shows a low percentage instead of "100 %".
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

STOP = set("a an the of in on at to for and or with without by from is are was were be this that these those it its "
           "showing shows show containing contain contains related about images image photos photo pictures picture "
           "videos video clips clip footage brochures brochure documents document pdf pdfs files file".split())
INTENT = {"video": {"video", "videos", "clip", "clips", "footage"},
          "pdf": {"brochure", "brochures", "pdf", "pdfs", "document", "documents", "flyer", "flyers", "catalog", "catalogue", "report", "manual"},
          "image": {"image", "images", "photo", "photos", "picture", "pictures", "pic", "pics"}}
RRF_K = 60
WEIGHTS = {"clip": 1.0, "text": 1.0, "kw": 0.5}
SECONDARY = 0.25        # weight of the non-best signals when combining
INTENT_BOOST = 1.3
KW_REL = 0.5            # keep keyword hits with BM25 >= 50 % of the best hit
CANDIDATES = 200


def detect_intent(q: str) -> set:
    words = set(re.findall(r"[a-z]+", q.lower()))
    return {t for t, vocab in INTENT.items() if words & vocab}


@dataclass
class Space:
    vids: np.ndarray
    aids: np.ndarray
    matrix: np.ndarray


EMPTY = Space(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 1), np.float32))


class Searcher:
    def __init__(self, settings, db, embedder):
        self.s, self.db, self.embedder = settings, db, embedder
        self._spaces: dict[str, Space] = {}
        self._sig = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------- in-memory matrices (reloaded on change)
    def _load(self):
        c = self.db.conn()
        sig = (tuple(c.execute("SELECT COUNT(*), COALESCE(MAX(id),0) FROM vectors").fetchone()), self.embedder.pipeline)
        if sig == self._sig:
            return
        with self._lock:
            if sig == self._sig:
                return
            spaces = {}
            for name in ("clip", "text"):
                rows = c.execute("SELECT v.id, v.asset_id, v.vec FROM vectors v JOIN assets a ON a.id=v.asset_id "
                                 "WHERE v.space=? AND a.pipeline=? AND a.status='ready' ORDER BY v.id",
                                 (name, self.embedder.pipeline)).fetchall()
                if not rows:
                    spaces[name] = EMPTY
                    continue
                m = np.frombuffer(b"".join(r[2] for r in rows), dtype="<f4").reshape(len(rows), -1).astype(np.float32)
                spaces[name] = Space(np.array([r[0] for r in rows], np.int64), np.array([r[1] for r in rows], np.int64), m)
            self._spaces, self._sig = spaces, sig

    # ----------------------------------------------------------- metadata filters
    def _allowed(self, types, exts, min_size, max_size, folder, after, before) -> np.ndarray:
        where = ["a.status='ready'", "a.pipeline=?"]
        args: list = [self.embedder.pipeline]
        if types:
            where.append(f"a.type IN ({','.join('?' * len(types))})"); args += list(types)
        if exts:
            where.append(f"a.ext IN ({','.join('?' * len(exts))})"); args += [e if e.startswith('.') else '.' + e for e in exts]
        if min_size is not None:
            where.append("a.size>=?"); args.append(int(min_size))
        if max_size is not None:
            where.append("a.size<=?"); args.append(int(max_size))
        if after is not None:
            where.append("a.indexed_ts>=?"); args.append(after)
        if before is not None:
            where.append("a.indexed_ts<=?"); args.append(before)
        fcond = ""
        if folder:
            fcond = " AND f.path LIKE ? ESCAPE '\\'"
            args = ["%" + folder.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_") + "%"] + args
        sql = (f"SELECT DISTINCT a.id FROM assets a JOIN files f ON f.asset_id=a.id AND f.present=1{fcond} "
               f"WHERE {' AND '.join(where)}")
        return np.array([r[0] for r in self.db.conn().execute(sql, args)], np.int64)

    # ----------------------------------------------------------- signals
    @staticmethod
    def _rank(space: Space, qv: np.ndarray, allowed: np.ndarray, floor: float, margin: float):
        if space.matrix.shape[0] == 0:
            return []
        sims = space.matrix @ qv
        ok = np.isin(space.aids, allowed)
        if not ok.any():
            return []
        top = float(sims[ok].max())
        if top < floor:                       # nothing in this signal is relevant enough
            return []
        cut = max(floor, top - margin)        # absolute floor AND relative margin
        idx = np.nonzero(ok & (sims >= cut))[0]
        out, seen = [], set()
        for i in idx[np.argsort(-sims[idx])]:
            a = int(space.aids[i])
            if a in seen:
                continue
            seen.add(a)
            out.append((a, int(space.vids[i]), float(sims[i])))
            if len(out) >= CANDIDATES:
                break
        return out

    def _keyword(self, q: str, allowed_set: set):
        toks = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if t not in STOP and len(t) > 1]
        if not toks:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))
        try:
            rows = self.db.conn().execute(
                "SELECT vtext.rowid AS vid, bm25(vtext) AS s, v.asset_id AS aid FROM vtext JOIN vectors v ON v.id=vtext.rowid "
                "WHERE vtext MATCH ? ORDER BY s LIMIT 600", (match,)).fetchall()
        except Exception:  # noqa: BLE001
            return []
        out, seen = [], set()
        for r in rows:
            if r["aid"] in allowed_set and r["aid"] not in seen:
                seen.add(r["aid"])
                out.append((r["aid"], r["vid"], -float(r["s"])))
        if out:
            top = out[0][2]
            out = [o for o in out if o[2] >= KW_REL * top]
        return out[:CANDIDATES]

    def _confidence(self, name: str, raw: float) -> float:
        if name == "kw":
            return min(1.0, raw / 12.0) * 0.8
        lo, hi = self.embedder.conf_clip if name == "clip" else self.embedder.conf_text
        return float(min(1.0, max(0.0, (raw - lo) / (hi - lo))))

    # ----------------------------------------------------------- calibration helper
    def raw_top(self, q: str) -> dict:
        """Best raw cosine per signal over the WHOLE index (no floors) - used by scripts/calibrate.py."""
        self._load()
        out = {}
        for name, enc in (("clip", self.embedder.encode_clip_text), ("text", self.embedder.encode_text)):
            sp = self._spaces.get(name, EMPTY)
            if sp.matrix.shape[0]:
                out[name] = float((sp.matrix @ enc([q])[0]).max())
        return out

    # ----------------------------------------------------------- main entry
    def search(self, q, *, types=None, exts=None, min_size=None, max_size=None, folder=None, after=None, before=None,
               limit=24, offset=0) -> dict:
        q = (q or "").strip()
        if not q:
            raise ValueError("empty query")
        t0 = time.perf_counter()
        self._load()
        em = self.embedder
        allowed = self._allowed(types, exts, min_size, max_size, folder, after, before)
        allowed_set = set(allowed.tolist())
        lists = {}
        if allowed.size:
            if self._spaces["clip"].matrix.shape[0]:
                lists["clip"] = self._rank(self._spaces["clip"], em.encode_clip_text([q])[0], allowed, em.min_clip, em.clip_margin)
            if self._spaces["text"].matrix.shape[0]:
                lists["text"] = self._rank(self._spaces["text"], em.encode_text([q])[0], allowed, em.min_text, em.text_margin)
            lists["kw"] = self._keyword(q, allowed_set)

        acc: dict[int, dict] = {}
        for name, lst in lists.items():
            for rank, (aid, vid, raw) in enumerate(lst):
                e = acc.setdefault(aid, {"contrib": {}, "ev": {}, "conf": {}, "vids": {}})
                e["contrib"][name] = WEIGHTS[name] / (RRF_K + rank + 1)
                e["ev"][name] = round(raw, 3)
                e["conf"][name] = self._confidence(name, raw)
                e["vids"][name] = vid
        for e in acc.values():
            cs = list(e["contrib"].values())
            e["score"] = max(cs) + SECONDARY * (sum(cs) - max(cs))
            e["confidence"] = max(e["conf"].values())
            e["best_sig"] = max(e["contrib"], key=e["contrib"].get)

        c = self.db.conn()
        ids = list(acc)
        types_of = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            for r in c.execute(f"SELECT id,type FROM assets WHERE id IN ({','.join('?' * len(part))})", part):
                types_of[r["id"]] = r["type"]
        intent = detect_intent(q) if not types else set()
        if intent:
            for aid, e in acc.items():
                if types_of.get(aid) in intent:
                    e["score"] *= INTENT_BOOST
        ranked = sorted(acc.items(), key=lambda kv: -kv[1]["score"])
        facets = {}
        for aid, _ in ranked:
            facets[types_of.get(aid)] = facets.get(types_of.get(aid), 0) + 1
        top = ranked[0][1]["score"] if ranked else 1.0
        results = [self._hydrate(aid, e, top) for aid, e in ranked[offset:offset + limit]]
        return {"query": q, "total": len(ranked), "offset": offset, "limit": limit, "intent": sorted(intent),
                "facets": facets, "signals": {k: len(v) for k, v in lists.items()},
                "took_ms": round((time.perf_counter() - t0) * 1000, 1), "results": results}

    def _vec(self, vid):
        v = self.db.conn().execute("SELECT kind,ref,text,thumb FROM vectors WHERE id=?", (vid,)).fetchone()
        return v

    def _hydrate(self, aid, e, top):
        c = self.db.conn()
        a = c.execute("SELECT * FROM assets WHERE id=?", (aid,)).fetchone()
        files = c.execute("SELECT id,path,folder,name FROM files WHERE asset_id=? AND present=1 ORDER BY path", (aid,)).fetchall()
        v = self._vec(e["vids"][e["best_sig"]])
        tv = self._vec(e["vids"].get("text") or e["vids"].get("kw")) if (e["vids"].get("text") or e["vids"].get("kw")) else None

        def snip(x):
            return (x["text"][:300] + ("…" if len(x["text"]) > 300 else "")) if x and x["text"] else None
        return {
            "asset_id": aid, "type": a["type"], "ext": a["ext"], "size": a["size"], "width": a["width"], "height": a["height"],
            "duration": a["duration"], "pages": a["pages"], "mime": a["mime"],
            "name": files[0]["name"] if files else None,
            "score": round(e["score"] / top, 4),              # ordering score (relative)
            "confidence": round(e["confidence"], 4),          # absolute match strength 0..1
            "evidence": e["ev"],
            "match": {"kind": v["kind"], "ref": v["ref"], "thumb": v["thumb"] or a["thumb"], "snippet": snip(v)} if v else None,
            "text_match": {"kind": tv["kind"], "ref": tv["ref"], "snippet": snip(tv)} if tv and tv["text"] else None,
            "thumb": a["thumb"],
            "files": [dict(f) for f in files],
            "duplicates": max(len(files) - 1, 0),
            "warnings": json.loads(a["warnings"] or "[]"),
            "meta": json.loads(a["meta"] or "{}"),
            "indexed_ts": a["indexed_ts"],
        }
