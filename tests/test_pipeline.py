"""End-to-end tests of ingest -> incremental re-index -> search, using the fake model backend
(no downloads). They exercise the reliability requirements: dedupe, corrupt/unsupported files,
AI failure, change detection, crash recovery, missing files."""
import os
import time

import cv2
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dam.api import create_app
from dam.config import Settings
from dam.db import Database
from dam.embedders import FakeEmbedder
from dam.indexer import Indexer
from dam.search import Searcher


def solid(path, rgb, size=(64, 48)):
    Image.new("RGB", size, rgb).save(path)


def make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        p = doc.new_page()
        p.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=11)
    doc.save(path)
    doc.close()


def make_video(path, color_bgr_list, fps=10, seconds_each=2):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 64))
    for col in color_bgr_list:
        frame = np.full((64, 96, 3), col, np.uint8)
        for _ in range(fps * seconds_each):
            w.write(frame)
    w.release()


@pytest.fixture()
def env(tmp_path):
    media, data = tmp_path / "media", tmp_path / "index"
    (media / "a").mkdir(parents=True)
    (media / "b").mkdir()
    s = Settings(media_dir=media, data_dir=data, model_backend="fake", workers=3,
                 video_min_interval=2, video_max_frames=6)
    emb = FakeEmbedder()
    db = Database(s.db_path)
    return s, media, db, emb


def run(s, db, emb, **kw):
    ix = Indexer(s, db, emb)
    ix.start(background=False, **kw)
    return ix


def populate(media):
    solid(media / "a" / "red.png", (250, 5, 5))
    solid(media / "a" / "blue.jpg", (5, 5, 250))
    shutil.copy(media / "a" / "red.png", media / "b" / "red_copy.png")          # duplicate content
    make_pdf(media / "a" / "solar.pdf", ["Residential solar panel installation guide for homeowners. " * 5,
                                           "Maintenance of inverter and battery storage. " * 5])
    make_pdf(media / "b" / "pasta.pdf", ["Classic carbonara recipe with eggs pecorino and guanciale. " * 5])
    make_video(media / "a" / "clip.mp4", [(0, 0, 255), (255, 0, 0)])             # red then blue (BGR)
    (media / "a" / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 this is not a jpeg")
    (media / "a" / "notes.txt").write_text("hello")
    (media / "a" / "empty.png").write_bytes(b"")
    (media / "a" / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")


import shutil  # noqa: E402


def states(db):
    return {os.path.basename(r["path"]): (r["state"], r["error"]) for r in
            db.conn().execute("SELECT path,state,error FROM files WHERE present=1")}


def test_full_flow_and_failure_handling(env):
    s, media, db, emb = env
    populate(media)
    ix = run(s, db, emb)
    st = states(db)
    assert st["red.png"][0] == st["blue.jpg"][0] == st["solar.pdf"][0] == st["clip.mp4"][0] == "done"
    assert st["red_copy.png"][0] == "done"
    assert st["broken.jpg"][0] == "failed" and "corrupt" in st["broken.jpg"][1]
    assert st["empty.png"][0] == "failed" and "empty" in st["empty.png"][1]
    assert st["broken.pdf"][0] == "failed"
    assert st["notes.txt"][0] == "unsupported"
    # duplicates share one asset; only one was analysed
    assert ix.stats.reused >= 1
    n_assets = db.conn().execute("SELECT COUNT(*) FROM assets WHERE status='ready'").fetchone()[0]
    assert n_assets == 5   # red, blue, solar, pasta, clip


def test_incremental_rerun_does_no_work(env):
    s, media, db, emb = env
    populate(media)
    run(s, db, emb)
    ix2 = run(s, db, emb)
    assert ix2.stats.queued == 0 and ix2.stats.indexed == 0 and ix2.stats.reused == 0
    assert ix2.stats.unchanged >= 8


def test_changed_and_new_files_only(env):
    s, media, db, emb = env
    populate(media)
    run(s, db, emb)
    time.sleep(0.05)
    solid(media / "a" / "blue.jpg", (5, 250, 5))          # content changed
    solid(media / "b" / "new.png", (250, 250, 5))         # new
    ix = run(s, db, emb)
    assert ix.stats.queued == 2 and ix.stats.indexed == 2
    # old blue asset is garbage-collected, no orphan vectors remain
    assert db.conn().execute("SELECT COUNT(*) FROM assets a WHERE NOT EXISTS (SELECT 1 FROM files f WHERE f.asset_id=a.id)").fetchone()[0] == 0


def test_touch_without_content_change_skips_ai(env):
    s, media, db, emb = env
    populate(media)
    run(s, db, emb)
    p = media / "a" / "red.png"
    os.utime(p, (time.time() + 100, time.time() + 100))
    ix = run(s, db, emb)
    assert ix.stats.queued == 1 and ix.stats.indexed == 0 and ix.stats.reused == 1


def test_ai_failure_is_isolated_and_retryable(env):
    s, media, db, emb = env
    solid(media / "a" / "red.png", (250, 5, 5))
    make_pdf(media / "a" / "t.pdf", ["plain text document about gardening and tomatoes. " * 4])
    import dam.indexer as im
    im.time.sleep = lambda *_: None            # don't wait on retry backoff in tests
    ix = run(s, db, FakeEmbedder(fail_on_images=True))
    st = states(db)
    assert st["red.png"][0] == "failed" and "AI processing failed" in st["red.png"][1]
    assert st["t.pdf"][0] == "failed"           # PDF also needs page images in this fake setup
    # fixed model + retry_failed re-processes them
    run(s, db, FakeEmbedder(), retry_failed=True)
    assert all(v[0] == "done" for k, v in states(db).items())


def test_crash_recovery_resets_processing(env):
    s, media, db, emb = env
    solid(media / "a" / "red.png", (250, 5, 5))
    run(s, db, emb)
    db.conn().execute("UPDATE files SET state='processing'")
    Indexer(s, db, emb).recover()
    assert db.conn().execute("SELECT state FROM files").fetchone()[0] == "pending"


def test_search_visual_text_and_filters(env):
    s, media, db, emb = env
    populate(media)
    run(s, db, emb)
    sr = Searcher(s, db, emb)

    r = sr.search("a red picture")
    top = r["results"][0]
    assert top["type"] == "image" and top["name"].startswith("red")
    assert top["duplicates"] == 1 and len(top["files"]) == 2     # both locations reported

    r = sr.search("blue video")                                   # intent boost + visual frame match
    assert r["results"][0]["type"] == "video"
    assert r["results"][0]["match"]["kind"] == "frame" and r["results"][0]["match"]["ref"] >= 2

    r = sr.search("solar panel installation for homeowners")
    assert r["results"][0]["name"] == "solar.pdf"
    assert r["results"][0]["match"]["snippet"]

    r = sr.search("carbonara", types=["pdf"])
    assert [x["name"] for x in r["results"]] == ["pasta.pdf"]
    assert sr.search("carbonara", types=["image"])["results"] == []
    in_b = sr.search("red", folder=os.sep + "b" + os.sep)
    assert in_b["total"] == 1 and any("/b/" in f["path"].replace(os.sep, "/") for f in in_b["results"][0]["files"])
    assert sr.search("zzzzqqqq nonsense")["total"] == 0
    assert sr.search("red", min_size=10**9)["total"] == 0


def test_missing_file_disappears_from_search_but_index_survives(env):
    s, media, db, emb = env
    populate(media)
    run(s, db, emb)
    os.remove(media / "b" / "pasta.pdf")
    run(s, db, emb)
    assert Searcher(s, db, emb).search("carbonara")["total"] == 0
    # empty/unmounted media dir must NOT wipe the index
    for p in list(media.rglob("*")):
        if p.is_file():
            p.unlink()
    run(s, db, emb)
    assert db.conn().execute("SELECT COUNT(*) FROM files WHERE present=1").fetchone()[0] > 0


def test_api_end_to_end(env):
    s, media, db, emb = env
    populate(media)
    app = create_app(s, emb)
    c = TestClient(app)
    assert c.post("/api/index/start").status_code == 200
    app.state.indexer.wait(30)
    stt = c.get("/api/index/status").json()
    assert stt["run"]["phase"] == "done" and stt["run"]["failed"] >= 3
    res = c.get("/api/search", params={"q": "solar panels", "types": "pdf"}).json()
    assert res["results"][0]["name"] == "solar.pdf"
    f = res["results"][0]["files"][0]
    assert c.get(f"/api/files/{f['id']}/content").headers["content-type"] == "application/pdf"
    r = c.get(f"/api/files/{f['id']}/content", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and len(r.content) == 10      # video seeking works
    assert c.get("/thumbs/" + res["results"][0]["thumb"]).status_code == 200
    fails = c.get("/api/files").json()
    assert {x["name"] for x in fails} >= {"broken.jpg", "notes.txt", "empty.png", "broken.pdf"}
    assert c.get("/api/stats").json()["duplicate_groups"] == 1
    assert len(c.get("/api/duplicates").json()) == 1
    assert c.get("/api/search", params={"q": ""}).status_code == 422


def test_negative_query_and_absolute_confidence(env):
    s, media, db, emb = env
    populate(media)
    run(s, db, emb)
    sr = Searcher(s, db, emb)
    assert sr.search("magenta")["total"] == 0                 # nothing close enough -> nothing returned (no "least bad" results)
    top = sr.search("a red picture")["results"][0]
    assert top["confidence"] > 0.99                           # strong absolute match
    weak = sr.search("solar", types=["pdf"])["results"][0]    # relative score is always 1.0 for the top hit ...
    assert weak["score"] == 1.0 and weak["confidence"] < 0.9  # ... but absolute confidence is not


def test_relative_margin_cuts_the_tail(env):
    s, media, db, emb = env
    solid(media / "a" / "red.png", (250, 5, 5))
    solid(media / "a" / "nearly_red.png", (240, 40, 40))      # cosine ~0.98 to "red": passes the floor
    run(s, db, emb)
    emb.clip_margin = 0.5
    assert Searcher(s, db, emb).search("red")["total"] == 2
    emb.clip_margin = 0.01                                    # only hits within 0.01 of the best survive
    assert Searcher(s, db, emb).search("red")["total"] == 1


def test_pdf_hit_carries_extracted_text_and_facets(env):
    s, media, db, emb = env
    populate(media)
    run(s, db, emb)
    r = Searcher(s, db, emb).search("solar panel installation for homeowners")
    top = r["results"][0]
    assert top["text_match"]["snippet"] and top["text_match"]["ref"] >= 1
    assert r["facets"].get("pdf", 0) >= 1
