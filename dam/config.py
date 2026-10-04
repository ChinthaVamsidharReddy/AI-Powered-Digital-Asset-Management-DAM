from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _b(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    media_dir: Path = Path("./data/media")
    data_dir: Path = Path("./data/index")
    model_backend: str = "clip"
    clip_model: str = "clip-ViT-B-32"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    workers: int = 4
    video_max_frames: int = 24
    video_min_interval: float = 8.0
    video_dedupe_threshold: float = 0.97
    video_time_budget: float = 180.0
    pdf_max_text_pages: int = 300
    pdf_max_render_pages: int = 8
    frame_max_side: int = 512
    thumb_size: int = 320
    max_image_pixels: int = 120_000_000
    enable_asr: bool = False
    asr_model: str = "tiny.en"
    clip_min_score: float = 0.20
    text_min_score: float = 0.30
    clip_margin: float = 0.05
    text_margin: float = 0.10
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "dam.sqlite3"

    @property
    def thumbs_dir(self) -> Path:
        return Path(self.data_dir) / "thumbs"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        e = os.environ.get
        return cls(
            media_dir=Path(e("MEDIA_DIR", "./data/media")),
            data_dir=Path(e("DATA_DIR", "./data/index")),
            model_backend=e("MODEL_BACKEND", "clip"),
            clip_model=e("CLIP_MODEL", "clip-ViT-B-32"),
            text_model=e("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            workers=int(e("WORKERS", "4")),
            video_max_frames=int(e("VIDEO_MAX_FRAMES", "24")),
            video_min_interval=float(e("VIDEO_MIN_INTERVAL", "8")),
            video_dedupe_threshold=float(e("VIDEO_DEDUPE_THRESHOLD", "0.97")),
            video_time_budget=float(e("VIDEO_TIME_BUDGET_SEC", "180")),
            pdf_max_text_pages=int(e("PDF_MAX_TEXT_PAGES", "300")),
            pdf_max_render_pages=int(e("PDF_MAX_RENDER_PAGES", "8")),
            enable_asr=_b("ENABLE_ASR", False),
            asr_model=e("ASR_MODEL", "tiny.en"),
            clip_min_score=float(e("CLIP_MIN_SCORE", "0.20")),
            text_min_score=float(e("TEXT_MIN_SCORE", "0.30")),
            clip_margin=float(e("CLIP_MARGIN", "0.05")),
            text_margin=float(e("TEXT_MARGIN", "0.10")),
            host=e("HOST", "127.0.0.1"),
            port=int(e("PORT", "8000")),
        )
