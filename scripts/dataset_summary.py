#!/usr/bin/env python3
"""Print (and optionally save) a dataset summary: counts and sizes per type and per sub-folder."""
import argparse
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dam.extractors import type_for_ext  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/media")
    ap.add_argument("--out", default=None, help="write markdown here, e.g. docs/DATASET.md")
    a = ap.parse_args()
    by_type = collections.defaultdict(lambda: [0, 0])
    by_folder = collections.defaultdict(lambda: [0, 0])
    for dp, _, fns in os.walk(a.dir):
        for fn in fns:
            if fn.startswith("."):
                continue
            p = Path(dp) / fn
            sz = p.stat().st_size
            t = type_for_ext(p.suffix) or "unsupported/other"
            if fn in ("manifest.json", "ATTRIBUTION.csv"):
                continue
            by_type[t][0] += 1; by_type[t][1] += sz
            top = str(Path(dp).relative_to(a.dir)).split(os.sep)[:2]
            k = "/".join(top) or "."
            by_folder[k][0] += 1; by_folder[k][1] += sz
    lines = ["# Dataset summary", "", f"Source folder: `{a.dir}`", "", "| Type | Files | Size |", "|---|---:|---:|"]
    tot = [0, 0]
    for t, (n, b) in sorted(by_type.items()):
        lines.append(f"| {t} | {n} | {b / 1e9:.2f} GB |"); tot[0] += n; tot[1] += b
    lines.append(f"| **Total** | **{tot[0]}** | **{tot[1] / 1e9:.2f} GB** |")
    lines += ["", "| Folder | Files | Size |", "|---|---:|---:|"]
    for k, (n, b) in sorted(by_folder.items()):
        lines.append(f"| {k} | {n} | {b / 1e6:.0f} MB |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
