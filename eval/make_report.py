#!/usr/bin/env python3
"""Generate the technical test report straight from the LIVE search results (nothing typed by hand).

For every test case it records: query, #results, response time, and for the top-N ranked results the
modality, match strength, filename, size, video duration + matching timestamp, PDF page count +
matching page + extracted matching text.

    python eval/make_report.py --top 15 --out docs/TEST_REPORT.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dam.config import Settings  # noqa: E402
from dam.db import Database  # noqa: E402
from dam.embedders import make_embedder  # noqa: E402
from dam.search import Searcher  # noqa: E402


def size(b):
    return f"{b / 1e6:.1f} MB" if b >= 1e6 else f"{b / 1e3:.0f} KB" if b >= 1e3 else f"{b} B"


def mmss(sec):
    sec = int(round(sec or 0))
    return f"{sec // 60}:{sec % 60:02d}"


def line(i, r):
    m, t = r["match"] or {}, r.get("text_match") or {}
    parts = [f"**{r['name']}**", f"{r['type'].capitalize() if r['type'] != 'pdf' else 'PDF'} **{round(r['confidence'] * 100)}%**", size(r["size"])]
    if r["type"] == "video":
        parts += [mmss(r["duration"]), f"match at **{mmss(m.get('ref'))}**" if m.get("kind") == "frame" else "match n/a"]
    if r["type"] == "pdf":
        parts.append(f"{r['pages']} page{'s' if r['pages'] != 1 else ''}")
        if m.get("kind") == "page":
            parts.append(f"visual match on page {int(m['ref'])}")
        if t.get("snippet"):
            parts.append(f"text on page {int(t['ref']) if t.get('ref') else '-'}: \"{' '.join(t['snippet'].split())[:160]}…\"")
        elif m.get("snippet"):
            parts.append(f"text on page {int(m['ref']) if m.get('ref') else '-'}: \"{' '.join(m['snippet'].split())[:160]}…\"")
    if r["duplicates"]:
        parts.append(f"{r['duplicates'] + 1} copies")
    return f"{i}. " + " — ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/report_cases.json")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="docs/TEST_REPORT.md")
    a = ap.parse_args()
    s = Settings.from_env()
    sr = Searcher(s, Database(s.db_path), make_embedder(s))
    sr.search("warm up")                                   # load models/matrices so timings are steady-state
    cases = json.loads(Path(a.cases).read_text(encoding="utf-8"))
    out = [f"# Search module - technical test report\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} from live results · pipeline `{sr.embedder.pipeline}` · "
           f"floors clip={s.clip_min_score}/text={s.text_min_score}, margins clip={s.clip_margin}/text={s.text_margin}\n",
           "Match strength is the *absolute* strength of the best signal (not relative to the top hit). "
           "Response time is the warm, in-process search time.\n"]
    n = 0
    for section, items in cases.items():
        out.append(f"\n# {section}\n")
        for it in items:
            q, types = (it, None) if isinstance(it, str) else (it["query"], it.get("types"))
            n += 1
            if q.startswith("<<<"):
                out.append(f"## Test Case {n} — query text not provided\n\nFill in `eval/report_cases.json` and re-run.\n")
                continue
            res = sr.search(q, types=types, limit=a.top)
            hdr = "Filter Test" if section == "Test Filters" else "Test Case"
            out.append(f"## {hdr} {n} — \"{q}\"\n")
            if section == "Test Filters":
                out.append(f"- Filter: **{', '.join(types) if types else 'No filter'}**")
            out += [f"- Search response: **{res['total']} results** (by type: {res['facets']})",
                    f"- Response time: **{res['took_ms']} ms**", "", "Ranked results:\n"]
            out += [line(i, r) for i, r in enumerate(res["results"], 1)] or ["_no results_"]
            out.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
