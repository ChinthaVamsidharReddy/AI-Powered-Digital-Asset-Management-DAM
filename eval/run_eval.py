#!/usr/bin/env python3
"""Search-quality evaluation.

Ground truth: for each query, a file is 'relevant' if its Wikimedia metadata (title, categories,
description - stored in manifest.json by download_dataset.py) matches `relevant_regex` AND its type
is in `expect_types`. The retrieval system never sees that metadata, so the labels are independent
of the system under test. They are WEAK labels (Commons descriptions are incomplete), therefore the
report also lists the top results so a human can judge relevance - fill the 'Manual verdict' column
and the notes section; that manual review is what the final EVALUATION.md should be based on.

    python eval/run_eval.py --manifest data/media/manifest.json --k 10 --out docs/EVAL_RESULTS.md
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dam.config import Settings  # noqa: E402
from dam.db import Database  # noqa: E402
from dam.embedders import make_embedder  # noqa: E402
from dam.search import Searcher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="eval/queries.json")
    ap.add_argument("--manifest", default="data/media/manifest.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="docs/EVAL_RESULTS.md")
    a = ap.parse_args()

    s = Settings.from_env()
    db = Database(s.db_path)
    searcher = Searcher(s, db, make_embedder(s))
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8")) if Path(a.manifest).exists() else {}
    root = s.media_dir.resolve()
    meta_text = {}
    for rel, m in manifest.items():
        meta_text[str((root / rel).resolve())] = (m["title"] + " " + " ".join(m["categories"]) + " " + m["description"]).lower()

    indexed = {r["path"]: r["type"] for r in db.conn().execute(
        "SELECT f.path, a.type FROM files f JOIN assets a ON a.id=f.asset_id WHERE f.present=1 AND a.status='ready'")}
    queries = json.loads(Path(a.queries).read_text(encoding="utf-8"))

    rows, md, agg = [], [], {"p5": [], "pk": [], "rec": [], "mrr": []}
    for q in queries:
        rx = re.compile(q["relevant_regex"], re.I)
        rel_paths = {p for p, t in indexed.items() if t in q["expect_types"] and rx.search(meta_text.get(os.path.abspath(p), meta_text.get(p, "")))}
        res = searcher.search(q["query"], limit=a.k)
        got = [r for r in res["results"]]
        hits = [any(f["path"] in rel_paths or os.path.abspath(f["path"]) in rel_paths for f in r["files"]) for r in got]
        p5 = sum(hits[:5]) / 5
        pk = sum(hits) / a.k
        rec = (sum(hits) / len(rel_paths)) if rel_paths else None
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        negative = not q["expect_types"]
        if not negative:
            agg["p5"].append(p5); agg["pk"].append(pk); agg["mrr"].append(1 / first if first else 0)
            if rec is not None:
                agg["rec"].append(rec)
        md.append(f"### {q['id']} - \"{q['query']}\"\n")
        md.append(f"*Looking for:* {q['intent']}  \n*Expected assets:* `{q['expected_hint']}` ({len(rel_paths)} labelled relevant in index)  \n")
        if negative:
            md.append(f"*Negative control:* returned **{res['total']}** results (ideal: 0 or very few).  \n")
        else:
            md.append(f"*Auto metrics:* P@5 = {p5:.2f}, P@{a.k} = {pk:.2f}, recall@{a.k} = {'n/a' if rec is None else f'{rec:.2f}'}, first relevant rank = {first or '-'}  \n")
        md.append(f"*Latency:* {res['took_ms']} ms  \n\n| # | Type | File | Match | Score | Auto-label | Manual verdict |\n|---|---|---|---|---|---|---|")
        for i, (r, h) in enumerate(zip(got, hits), 1):
            m = r["match"] or {}
            md.append(f"| {i} | {r['type']} | `{r['files'][0]['path'].replace(str(root), '')}` | {m.get('kind', '')} {m.get('ref', '') or ''} | {r['score']:.2f} | {'✔' if h else '✘'} |  |")
        md.append("\n**Notes / failure analysis:** _(fill in after looking at the results)_\n")

    def avg(x): return f"{sum(x) / len(x):.2f}" if x else "n/a"
    head = [f"# Search evaluation results\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')} · index pipeline `{searcher.embedder.pipeline}` · {len(indexed)} indexed assets\n",
            "| Metric (non-negative queries) | Value |", "|---|---|",
            f"| Mean P@5 | {avg(agg['p5'])} |", f"| Mean P@{a.k} | {avg(agg['pk'])} |",
            f"| Mean recall@{a.k} | {avg(agg['rec'])} |", f"| MRR | {avg(agg['mrr'])} |\n",
            "> Auto-labels come from Commons metadata and are noisy; the *Manual verdict* column is the source of truth.\n"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(head + md), encoding="utf-8")
    print("\n".join(head)); print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
