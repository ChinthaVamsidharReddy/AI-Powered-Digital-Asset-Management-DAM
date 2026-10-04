#!/usr/bin/env python3
"""Pick CLIP_MIN_SCORE / TEXT_MIN_SCORE from YOUR data (needs the real models and a finished index).

Prints the best raw cosine per signal for (a) queries that SHOULD match and (b) queries that should NOT.
A good floor sits above every negative and below every positive. Copy the recommendation into .env.

    python scripts/calibrate.py
    python scripts/calibrate.py --negatives "spaceship on Mars" "your own nonsense query"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dam.config import Settings  # noqa: E402
from dam.db import Database  # noqa: E402
from dam.embedders import make_embedder  # noqa: E402
from dam.search import Searcher  # noqa: E402

NEG = ["spaceship on Mars", "underwater volcano erupting at night", "a purple elephant playing the violin",
       "medieval knights jousting"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/queries.json")
    ap.add_argument("--negatives", nargs="*", default=NEG)
    a = ap.parse_args()
    s = Settings.from_env()
    sr = Searcher(s, Database(s.db_path), make_embedder(s))
    pos = [q["query"] for q in json.loads(Path(a.queries).read_text(encoding="utf-8")) if q["expect_types"]]
    P = {q: sr.raw_top(q) for q in pos}
    N = {q: sr.raw_top(q) for q in a.negatives}
    print(f"{'query':<55} {'clip':>6} {'text':>6}")
    for title, D in (("SHOULD match", P), ("should NOT match", N)):
        print(f"--- {title}")
        for q, v in D.items():
            print(f"{q[:54]:<55} {v.get('clip', float('nan')):>6.3f} {v.get('text', float('nan')):>6.3f}")
    print()
    for sig, env in (("clip", "CLIP_MIN_SCORE"), ("text", "TEXT_MIN_SCORE")):
        pv = sorted(v[sig] for v in P.values() if sig in v)
        nv = [v[sig] for v in N.values() if sig in v]
        if not pv or not nv:
            continue
        floor = max(nv) + 0.01
        # positives are for different modalities, so 'clip' positives include PDF-only queries that legitimately score low
        weak = [round(x, 3) for x in pv if x < floor]
        print(f"{env}={floor:.2f}   (highest negative {max(nv):.3f}; {len(weak)} of {len(pv)} positive queries fall below this: {weak})")
    print("\nNote: text-only queries (PDF topics) legitimately score low on 'clip' and image queries low on 'text'; "
          "judge each signal on the queries of its own modality.")


if __name__ == "__main__":
    main()
