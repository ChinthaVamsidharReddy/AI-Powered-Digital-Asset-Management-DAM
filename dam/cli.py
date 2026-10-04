from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .config import Settings


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dam", description="AI Digital Asset Management")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="start the web UI + API")
    ix = sub.add_parser("index", help="scan MEDIA_DIR and index new/changed files (resumable)")
    ix.add_argument("--retry-failed", action="store_true", help="retry files that failed before")
    ix.add_argument("--force", action="store_true", help="re-process everything")
    sr = sub.add_parser("search", help="query from the terminal")
    sr.add_argument("query")
    sr.add_argument("-n", type=int, default=10)
    sr.add_argument("--types", default=None)
    sub.add_parser("stats", help="show index statistics")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings.from_env()

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("dam.api:create_app", factory=True, host=s.host, port=s.port, log_level="info")
        return

    from .db import Database
    from .embedders import make_embedder
    db, emb = Database(s.db_path), make_embedder(s)

    if a.cmd == "index":
        from .indexer import Indexer
        ix_ = Indexer(s, db, emb)
        ix_.start(retry_failed=a.retry_failed, force=a.force, background=True)
        try:
            while ix_.running:
                st = ix_.status()
                r = st["run"]
                print(f"\r[{r['phase']:<10}] scanned={r['scanned']} queued={r['queued']} done={r['processed']} "
                      f"indexed={r['indexed']} reused={r['reused']} failed={r['failed']} unsupported={r['unsupported']} "
                      f"eta={st['eta_sec']}s   ", end="", flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nCancelling (progress is saved; re-run to resume)...")
            ix_.cancel()
            ix_.wait()
        print("\n" + json.dumps(ix_.status()["run"], indent=2))
    elif a.cmd == "search":
        from .search import Searcher
        res = Searcher(s, db, emb).search(a.query, types=a.types.split(",") if a.types else None, limit=a.n)
        for i, r in enumerate(res["results"], 1):
            m = r["match"] or {}
            print(f"{i:>2}. [{r['type']:<5}] {r['score']:.2f} {r['files'][0]['path']}  ({m.get('kind')} @ {m.get('ref')}) {r['evidence']}")
        print(f"{res['total']} results in {res['took_ms']} ms")
    elif a.cmd == "stats":
        c = db.conn()
        print(json.dumps({"files": {r[0]: r[1] for r in c.execute("SELECT state,COUNT(*) FROM files WHERE present=1 GROUP BY state")},
                          "assets": {r[0]: r[1] for r in c.execute("SELECT type,COUNT(*) FROM assets WHERE status='ready' GROUP BY type")},
                          "vectors": {r[0]: r[1] for r in c.execute("SELECT space,COUNT(*) FROM vectors GROUP BY space")}}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
