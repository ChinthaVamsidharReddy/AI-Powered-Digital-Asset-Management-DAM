"""Per-type content extraction: metadata + (images to embed with CLIP) + (text to embed / search).

Video approach: no per-frame processing. We sample at most N frames, spread evenly over the duration
(never closer than `video_min_interval` seconds), skip blank frames, then the indexer drops
near-duplicate frames by embedding similarity. Optional ASR adds a searchable transcript.

PDF approach: per-page text -> overlapping chunks (text embedding + FTS keyword index) AND a handful
of rendered pages -> CLIP (so image-heavy brochures / scanned PDFs are still findable visually).
"""
from __future__ import annotations

import mimetypes
import os
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ogv", ".mpg", ".mpeg"}
PDF_EXTS = {".pdf"}


def type_for_ext(ext: str):
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in PDF_EXTS:
        return "pdf"
    return None


class UnsupportedFile(Exception):
    pass


class CorruptFile(Exception):
    pass


@dataclass
class VisualItem:
    kind: str            # image | frame | page
    ref: float | None    # seconds or 1-based page
    image: Image.Image


@dataclass
class TextItem:
    kind: str            # chunk | meta | transcript
    ref: float | None
    text: str


@dataclass
class Extracted:
    type: str
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    pages: int | None = None
    meta: dict = field(default_factory=dict)
    visuals: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# ------------------------------------------------------------------------------------------ images
def _to_rgb(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return im.convert("RGB")


def extract_image(path, s) -> Extracted:
    Image.MAX_IMAGE_PIXELS = s.max_image_pixels
    meta: dict = {}
    try:
        with Image.open(path) as im:
            meta["format"] = im.format
            width, height = im.size
            try:
                exif = im.getexif()
                cam = " ".join(str(x).strip() for x in (exif.get(271), exif.get(272)) if x)
                if cam:
                    meta["camera"] = cam
                taken = exif.get_ifd(0x8769).get(36867) or exif.get(306)
                if taken:
                    meta["taken"] = str(taken)
            except Exception:  # noqa: BLE001  (EXIF is best-effort)
                pass
            im.draft("RGB", (s.frame_max_side * 2, s.frame_max_side * 2))  # fast JPEG downscale
            im.load()                                                      # raises on truncation
            rgb = _to_rgb(ImageOps.exif_transpose(im))
    except Image.DecompressionBombError as e:
        raise CorruptFile(f"image too large / decompression bomb: {e}") from e
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise CorruptFile(f"{type(e).__name__}: {e}") from e
    rgb.thumbnail((s.frame_max_side, s.frame_max_side))
    return Extracted("image", mimetypes.guess_type(str(path))[0], width, height, meta=meta,
                     visuals=[VisualItem("image", None, rgb)])


# ------------------------------------------------------------------------------------------ video
def _sample_times(duration: float, max_frames: int, min_interval: float):
    n = max(1, min(max_frames, int(duration // max(min_interval, 0.1)) or 1))
    return [(i + 0.5) * duration / n for i in range(n)]


def _frame_to_pil(frame, max_side):
    import cv2
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img.thumbnail((max_side, max_side))
    return img


def _sequential_sample(cap, fps, interval, max_frames, deadline):
    """Fallback when seeking is unreliable / duration unknown: walk the stream, keep 1 frame per interval."""
    import cv2
    out, idx, step = [], 0, max(1, int(round(interval * (fps or 25.0))))
    while len(out) < max_frames and time.monotonic() < deadline:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                out.append((t, frame))
        idx += 1
    return out


def extract_video(path, s) -> Extracted:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise CorruptFile("cannot open video (corrupt container or unsupported codec)")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        nframes = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip() if fourcc else None
        duration = nframes / fps if fps > 0 and 0 < nframes < 1e9 else None
        warnings: list = []
        deadline = time.monotonic() + s.video_time_budget

        raw = []
        if duration:
            for t in _sample_times(duration, s.video_max_frames, s.video_min_interval):
                if time.monotonic() > deadline:
                    warnings.append("time budget exhausted - partial frame sampling")
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    raw.append((t, frame))
        if not raw:  # unknown duration or seeking failed
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            raw = _sequential_sample(cap, fps, s.video_min_interval, s.video_max_frames, deadline)
            if raw and not duration:
                duration = raw[-1][0] or None
        if not raw:
            raise CorruptFile("no decodable frames")

        items = [(t, _frame_to_pil(f, s.frame_max_side)) for t, f in raw]
        nonblank = [(t, im) for t, im in items if np.asarray(im.convert("L")).std() > 2.0]
        if nonblank:
            items = nonblank
        else:
            warnings.append("all sampled frames are blank")
        ex = Extracted("video", mimetypes.guess_type(str(path))[0], width or None, height or None,
                       duration=round(duration, 2) if duration else None,
                       meta={"fps": round(fps, 2) if fps else None, "codec": codec, "frames_sampled": len(items)},
                       visuals=[VisualItem("frame", round(t, 2), im) for t, im in items], warnings=warnings)
    finally:
        cap.release()

    if s.enable_asr:
        try:
            ex.texts.extend(transcribe(path, s))
        except Exception as e:  # noqa: BLE001  ASR is an enrichment; never fail the file for it
            ex.warnings.append(f"transcription failed: {type(e).__name__}: {e}")
    return ex


_asr_model = None
_asr_lock = threading.Lock()


def transcribe(path, s):
    global _asr_model
    with _asr_lock:
        if _asr_model is None:
            from faster_whisper import WhisperModel
            _asr_model = WhisperModel(s.asr_model, device="cpu", compute_type="int8")
        segments, _ = _asr_model.transcribe(str(path), vad_filter=True)
        segs = [(seg.start, seg.text.strip()) for seg in segments if seg.text.strip()]
    out, buf, start = [], "", 0.0
    for t, text in segs:
        if not buf:
            start = t
        buf += " " + text
        if len(buf) > 500:
            out.append(TextItem("transcript", round(start, 1), buf.strip()))
            buf = ""
    if buf.strip():
        out.append(TextItem("transcript", round(start, 1), buf.strip()))
    return out


# ------------------------------------------------------------------------------------------ pdf
_pdf_lock = threading.Lock()  # MuPDF is not thread-safe


def chunk_text(text: str, size: int = 900, overlap: int = 150):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if len(text) >= 20 else []
    chunks, i = [], 0
    while i < len(text):
        j = min(i + size, len(text))
        if j < len(text):
            k = text.rfind(". ", i + size // 2, j)
            if k != -1:
                j = k + 1
        chunks.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return [c for c in chunks if len(c) >= 20]


def extract_pdf(path, s) -> Extracted:
    import pymupdf
    with _pdf_lock:
        try:
            doc = pymupdf.open(str(path))
        except Exception as e:  # noqa: BLE001
            raise CorruptFile(f"cannot open PDF: {e}") from e
        try:
            if doc.needs_pass:
                raise UnsupportedFile("password-protected PDF")
            n = doc.page_count
            if n == 0:
                raise CorruptFile("PDF has no pages")
            md = {k: v for k, v in (doc.metadata or {}).items() if v and k in ("title", "author", "subject", "keywords", "creator", "producer", "creationDate")}
            ex = Extracted("pdf", "application/pdf", pages=n, meta=md)
            r0 = doc[0].rect
            ex.width, ex.height = int(r0.width), int(r0.height)

            head = ". ".join(md[k] for k in ("title", "subject", "keywords") if k in md)
            if len(head) >= 5:
                ex.texts.append(TextItem("meta", None, head))
            total_chars, n_chunks = 0, 0
            for pno in range(min(n, s.pdf_max_text_pages)):
                try:
                    txt = doc[pno].get_text("text")
                except Exception as e:  # noqa: BLE001
                    ex.warnings.append(f"page {pno + 1}: text extraction failed ({e})")
                    continue
                total_chars += len(txt)
                for c in chunk_text(txt):
                    if n_chunks < 600:
                        ex.texts.append(TextItem("chunk", pno + 1, c))
                        n_chunks += 1
            if total_chars < 50:
                ex.warnings.append("no extractable text (scanned/image-only?) - indexed visually only")
            if n > s.pdf_max_text_pages:
                ex.warnings.append(f"text indexed for first {s.pdf_max_text_pages} of {n} pages")

            k = min(n, s.pdf_max_render_pages)
            pages = sorted({int(round(x)) for x in np.linspace(0, n - 1, k)}) if k else []
            for pno in pages:
                try:
                    page = doc[pno]
                    z = s.frame_max_side / max(page.rect.width, page.rect.height, 1)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), colorspace=pymupdf.csRGB, alpha=False)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    ex.visuals.append(VisualItem("page", pno + 1, img))
                except Exception as e:  # noqa: BLE001
                    ex.warnings.append(f"page {pno + 1}: render failed ({e})")
            return ex
        finally:
            doc.close()


def extract(path, ext: str, s) -> Extracted:
    t = type_for_ext(ext)
    if t == "image":
        return extract_image(path, s)
    if t == "video":
        return extract_video(path, s)
    if t == "pdf":
        return extract_pdf(path, s)
    raise UnsupportedFile(f"unsupported file type '{ext}'")
