#!/usr/bin/env python3
"""Build the mixed-media dataset from Wikimedia Commons (free-licensed / public domain, no API key).

    python scripts/download_dataset.py --target-gb 6 --contact you@example.com

* Searches Commons per topic, filters by MIME type, downloads originals (size-capped), resumable
  (existing files are skipped), polite (serial + delay + User-Agent with contact info as Wikimedia requires).
* Writes data/media/manifest.json (title/categories/description/licence per file) - used ONLY by the
  evaluation script to derive ground truth. It is never used by the search itself.
* Writes data/media/ATTRIBUTION.csv with author + licence + source URL for every file.
* --edge-cases adds duplicates, a truncated JPEG, a zero-byte file, an unsupported .txt and a bad PDF
  under media/_edge_cases/ so the reliability features can be demonstrated.

NOTE: queries below are starting points - Commons search results vary; adjust TOPICS freely.
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import re
import shutil
import sys
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
IMG = {"image/jpeg", "image/png", "image/webp"}
# Commons reports Ogg video (.ogv) as "application/ogg" (audio .ogg/.oga share that MIME), so for that
# MIME we additionally require a video extension. MP4 is accepted when present.
VID = {"video/webm", "video/ogg", "application/ogg", "video/mpeg", "video/mp4"}
VIDEO_EXT = (".webm", ".ogv", ".ogg", ".mpg", ".mpeg", ".mp4")
# Extra CirrusSearch keyword per kind so results are not dominated by other file types.
SEARCH_SUFFIX = {"video": " filetype:video"}
PDF = {"application/pdf"}

# (kind, tag, search phrase, max files)
TOPICS = [
    ("image", "woman-cat", "woman with cat", 40),
    ("image", "living-room", "modern living room interior", 60),
    ("image", "kitchen", "modern kitchen interior", 40),
    ("image", "bedroom", "bedroom interior design", 30),
    ("image", "construction", "construction site workers", 60),
    ("image", "building-exterior", "residential apartment building exterior", 60),
    ("image", "portrait", "person portrait outdoors", 50),
    ("image", "street", "city street", 50),
    ("image", "landscape", "mountain lake landscape", 50),
    ("image", "food", "plated food dish", 40),
    ("image", "dogs", "dog park", 30),
    ("image", "cars", "car showroom", 30),
    ("video", "construction", "construction machinery excavator", 12),
    ("video", "interview", "interview speaking to camera", 12),
    ("video", "nature", "wildlife animals", 12),
    ("video", "city", "city timelapse traffic", 12),
    ("video", "cooking", "cooking demonstration", 8),
    ("video", "demo", "product demonstration", 8),
    ("pdf", "brochure", "brochure", 40),
    ("pdf", "real-estate", "real estate residential project brochure", 30),
    ("pdf", "catalog", "product catalogue", 30),
    ("pdf", "manual", "user manual instructions", 30),
    ("pdf", "report", "annual report", 30),
]


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).strip()


class Forbidden(Exception):
    """HTTP 403/404: retrying will not help."""


def check_contact(contact: str):
    c = contact.lower()
    if "example." in c or "you@" in c or "@" not in c and not c.startswith("http"):
        sys.exit("ERROR: --contact must be a REAL e-mail address or URL (e.g. your own e-mail or GitHub repo).\n"
                 "Wikimedia's file servers return 403 for placeholder/anonymous User-Agents.\n"
                 "See https://meta.wikimedia.org/wiki/User-Agent_policy")


class Client:
    def __init__(self, contact: str, delay: float, user_agent: str | None = None):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent or (
            f"DigitalAssetManagementAssignment/1.0 ({contact}) requests/{requests.__version__}")
        self.s.headers["Accept"] = "*/*"
        self.delay = delay

    def get(self, url, **kw):
        for attempt in range(5):
            try:
                r = self.s.get(url, timeout=60, **kw)
                if r.status_code in (403, 404):
                    raise Forbidden(f"HTTP {r.status_code} for {url.split('?')[0]}")
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt * 2)
                    continue
                r.raise_for_status()
                time.sleep(self.delay)
                return r
            except Forbidden:
                raise
            except requests.RequestException as e:
                if attempt == 4:
                    raise
                print(f"  retry ({e})", file=sys.stderr)
                time.sleep(2 ** attempt * 2)

    def search(self, phrase, mimes, limit, kind="image", rejected=None):
        cont, found = {}, 0
        phrase = phrase + SEARCH_SUFFIX.get(kind, "")
        while found < limit * 3:
            params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": phrase,
                      "gsrnamespace": 6, "gsrlimit": 50, "prop": "imageinfo|categories", "cllimit": "max",
                      "iiprop": "url|size|mime|extmetadata", **cont}
            d = self.get(API, params=params).json()
            for p in (d.get("query", {}).get("pages", {}) or {}).values():
                ii = (p.get("imageinfo") or [{}])[0]
                mime = ii.get("mime")
                title = p.get("title", "").lower()
                ok = mime in mimes
                if ok and kind == "video" and mime == "application/ogg":
                    ok = title.endswith(".ogv")          # exclude audio-only Ogg files
                if ok and kind == "video" and not title.endswith(VIDEO_EXT):
                    ok = False
                if ok:
                    found += 1
                    yield p, ii
                elif rejected is not None:
                    rejected[mime] += 1
            if "continue" not in d:
                return
            cont = d["continue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/media")
    ap.add_argument("--target-gb", type=float, default=6.0)
    ap.add_argument("--max-file-mb", type=float, default=80.0)
    ap.add_argument("--contact", required=True, help="e-mail or URL, put into the User-Agent (Wikimedia policy)")
    ap.add_argument("--delay", type=float, default=0.7)
    ap.add_argument("--kinds", default="image,video,pdf")
    ap.add_argument("--edge-cases", action="store_true")
    ap.add_argument("--user-agent", default=None, help="override the User-Agent string completely")
    ap.add_argument("--debug", action="store_true", help="print why results were rejected for every topic")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mf_path = out / "manifest.json"
    manifest = json.loads(mf_path.read_text()) if mf_path.exists() else {}
    attrib = out / "ATTRIBUTION.csv"
    new_attr = not attrib.exists()
    attr_f = open(attrib, "a", newline="", encoding="utf-8")
    aw = csv.writer(attr_f)
    if new_attr:
        aw.writerow(["file", "title", "author", "license", "source_url"])

    budget = int(a.target_gb * 1024 ** 3)
    used = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    if not a.user_agent:
        check_contact(a.contact)
    cl = Client(a.contact, a.delay, a.user_agent)
    forbidden_streak = 0
    kinds = set(a.kinds.split(","))

    for kind, tag, phrase, cap in TOPICS:
        if kind not in kinds or used >= budget:
            continue
        mimes = {"image": IMG, "video": VID, "pdf": PDF}[kind]
        ddir = out / kind / tag
        ddir.mkdir(parents=True, exist_ok=True)
        got = len(list(ddir.iterdir()))
        print(f"[{kind}/{tag}] '{phrase}'  ({used / 1e9:.2f}/{a.target_gb} GB)")
        rejected = collections.Counter()
        skipped_size = 0
        for page, ii in cl.search(phrase, mimes, cap * 3, kind, rejected):
            if got >= cap or used >= budget:
                break
            if ii["size"] > a.max_file_mb * 1024 * 1024 or ii["size"] < 5_000:
                skipped_size += 1
                continue
            title = page["title"].removeprefix("File:")
            fname = re.sub(r"[^\w.\-]+", "_", title)[:140]
            dest = ddir / fname
            rel = str(dest.relative_to(out))
            if dest.exists():
                continue
            try:
                r = cl.get(ii["url"].split("?")[0], stream=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
            except Forbidden as e:
                forbidden_streak += 1
                print(f"  skip {title}: {e}", file=sys.stderr)
                if forbidden_streak >= 3:
                    sys.exit("\nAborting: 3 consecutive 403/404 from upload.wikimedia.org.\n"
                             "-> Use a REAL contact: --contact your.real@email.com (placeholders are blocked), or pass\n"
                             "   --user-agent \"MyProject/1.0 (https://github.com/you/repo; you@real.com)\".\n"
                             "-> If it still fails you may be rate-limited/blocked by your network: wait ~30 min, raise --delay 2.")
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  skip {title}: {e}", file=sys.stderr)
                continue
            forbidden_streak = 0
            used += dest.stat().st_size
            got += 1
            em = ii.get("extmetadata", {})
            cats = [c["title"].removeprefix("Category:") for c in page.get("categories", [])]
            manifest[rel] = {"title": title, "kind": kind, "tag": tag, "query": phrase, "categories": cats,
                             "description": strip_tags(em.get("ImageDescription", {}).get("value", "")),
                             "license": em.get("LicenseShortName", {}).get("value", ""),
                             "author": strip_tags(em.get("Artist", {}).get("value", ""))}
            aw.writerow([rel, title, manifest[rel]["author"], manifest[rel]["license"], ii.get("descriptionurl", "")])
            mf_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            attr_f.flush()
            print(f"  + {rel}  {dest.stat().st_size / 1e6:.1f} MB")
        if a.debug or got == 0:
            print(f"  [debug] downloaded/present={got}; skipped by size={skipped_size} (limit {a.max_file_mb} MB); "
                  f"rejected MIME types={dict(rejected) or '-'}")

    if a.edge_cases:
        edge = out / "_edge_cases"
        edge.mkdir(exist_ok=True)
        imgs = [p for p in out.glob("image/**/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if imgs:
            shutil.copy(imgs[0], edge / ("dup_" + imgs[0].name))                  # duplicate content, other folder
            data = imgs[1 % len(imgs)].read_bytes()
            (edge / "truncated.jpg").write_bytes(data[: max(len(data) // 3, 100)])  # corrupted image
        (edge / "empty.png").write_bytes(b"")
        (edge / "notes.txt").write_text("unsupported file type for the demo\n")
        (edge / "broken.pdf").write_bytes(b"%PDF-1.7\nthis is not a real pdf\n")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nDone. {total / 1e9:.2f} GB in {out}. Next: python scripts/dataset_summary.py")


if __name__ == "__main__":
    main()
