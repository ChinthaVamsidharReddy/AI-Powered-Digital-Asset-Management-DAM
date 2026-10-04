"""Embedding backends.

* ClipEmbedder - real models. CLIP embeds images/frames/page renders AND the text query into one
  space (cross-modal search); a sentence-transformer embeds document text/transcripts.
* FakeEmbedder - tiny deterministic stand-in so the pipeline/tests run with no downloads.
"""
from __future__ import annotations

import hashlib
import re
import threading

import numpy as np


class EmbedderError(Exception):
    """Model inference failed (OOM, bad input, model not loadable...)."""


class BaseEmbedder:
    name = "base"
    min_clip = 0.0
    min_text = 0.0
    clip_margin = 0.05      # keep hits within this cosine distance of the best hit
    text_margin = 0.10
    conf_clip = (0.18, 0.32)   # cosine range mapped to 0..100 % "match strength"
    conf_text = (0.25, 0.60)

    @property
    def pipeline(self) -> str:
        return self.name

    def warmup(self) -> None:  # pragma: no cover
        pass

    def encode_images(self, imgs) -> np.ndarray: ...
    def encode_clip_text(self, texts) -> np.ndarray: ...
    def encode_text(self, texts) -> np.ndarray: ...


class ClipEmbedder(BaseEmbedder):
    def __init__(self, settings):
        self.s = settings
        self.name = f"clip:{settings.clip_model}|text:{settings.text_model}"
        self.min_clip = settings.clip_min_score
        self.min_text = settings.text_min_score
        self.clip_margin = settings.clip_margin
        self.text_margin = settings.text_margin
        self._clip = None
        self._txt = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # one inference at a time; decoding/IO stays parallel

    def _models(self):
        if self._clip is None or self._txt is None:
            with self._load_lock:
                if self._clip is None:
                    from sentence_transformers import SentenceTransformer
                    self._clip = SentenceTransformer(self.s.clip_model)
                if self._txt is None:
                    from sentence_transformers import SentenceTransformer
                    self._txt = SentenceTransformer(self.s.text_model)
        return self._clip, self._txt

    def warmup(self):
        self._models()

    def _enc(self, model, items, batch=16):
        try:
            with self._infer_lock:
                v = model.encode(items, batch_size=batch, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            raise EmbedderError(f"{type(e).__name__}: {e}") from e

    def encode_images(self, imgs):
        return self._enc(self._models()[0], list(imgs))

    def encode_clip_text(self, texts):
        texts = [" ".join(t.split()[:55]) for t in texts]  # CLIP context is 77 tokens
        return self._enc(self._models()[0], texts)

    def encode_text(self, texts):
        return self._enc(self._models()[1], list(texts), batch=32)


# ---------------------------------------------------------------- fake (tests / offline smoke runs)
_COLORS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "yellow": (1, 1, 0),
           "white": (1, 1, 1), "cyan": (0, 1, 1), "magenta": (1, 0, 1)}


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class FakeEmbedder(BaseEmbedder):
    """CLIP stand-in: image -> normalised mean RGB; text -> sum of colour-word vectors.
    Text stand-in: hashed bag-of-words (lexical, deterministic)."""
    name = "fake"
    min_clip = 0.9
    min_text = 0.15
    clip_margin = 0.5
    text_margin = 0.6
    conf_clip = (0.0, 1.0)
    conf_text = (0.0, 0.5)
    DIM = 256

    def __init__(self, fail_on_images: bool = False):
        self.fail_on_images = fail_on_images

    def encode_images(self, imgs):
        if self.fail_on_images:
            raise EmbedderError("simulated model failure")
        out = []
        for im in imgs:
            a = np.asarray(im.convert("RGB").resize((16, 16)), dtype=np.float32).reshape(-1, 3).mean(0)
            out.append(_norm(np.pad(a, (0, 5))))
        return np.stack(out)

    def encode_clip_text(self, texts):
        out = []
        for t in texts:
            v = np.zeros(8, dtype=np.float32)
            for w in re.findall(r"[a-z]+", t.lower()):
                if w in _COLORS:
                    v[:3] += _COLORS[w]
            out.append(_norm(v) if v.any() else np.full(8, 0.0, dtype=np.float32))
        return np.stack(out)

    def encode_text(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.DIM, dtype=np.float32)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.DIM] += 1.0
            out.append(_norm(v))
        return np.stack(out)


def make_embedder(settings) -> BaseEmbedder:
    if settings.model_backend == "fake":
        return FakeEmbedder()
    return ClipEmbedder(settings)
