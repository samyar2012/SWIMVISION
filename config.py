"""Central configuration for SwimVision.

All filesystem locations live here so no other module hard-codes a path.
Secrets (e.g. an optional AI-coach API key) are read from the environment /
a local ``.env`` file and are NEVER stored in source code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Load variables from .env (if the file exists). Missing .env is fine.
load_dotenv(PROJECT_ROOT / ".env")

# --- Folders (all git-ignored except for .gitkeep placeholders) -------------
UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"      # raw user videos (private!)
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"      # annotated videos, clips, previews
DATA_DIR: Path = PROJECT_ROOT / "data"            # saved analyses (JSON)
ANALYSES_DIR: Path = DATA_DIR / "analyses"
PREVIEWS_DIR: Path = OUTPUTS_DIR / "previews"     # browser-friendly video copies
ANNOTATED_DIR: Path = OUTPUTS_DIR / "annotated"   # videos with tracking overlays
TRACKING_DIR: Path = DATA_DIR / "tracking"        # per-frame tracking tables (CSV)
MODELS_DIR: Path = PROJECT_ROOT / "models"        # downloaded pretrained weights

# --- Detection / tracking defaults (Phase 2) --------------------------------
DEFAULT_YOLO_MODEL: str = "yolo11s"               # small + accurate enough; see README
YOLO_MODEL_CHOICES: tuple[str, ...] = ("yolo11n", "yolo11s", "yolo11m", "yolo11l")
VIT_TRACKER_URL: str = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
)
VIT_TRACKER_PATH: Path = MODELS_DIR / "vittrack.onnx"
SAM2_MODEL_PATH: Path = MODELS_DIR / "sam2.1_b.pt"  # Segment Anything 2.1 (base), ~155 MB
TARGET_ANALYSIS_FPS: float = 30.0                 # we never need to analyse faster than this
ANNOTATED_MAX_HEIGHT: int = 720                   # annotated video is downscaled to this

# --- Video handling ---------------------------------------------------------
# Containers we accept in the uploader. Whether a *specific* file can be
# decoded is checked separately by OpenCV after upload.
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = ("mp4", "mov", "avi", "m4v", "mkv")

# Longest preview side we re-encode to when the original is not browser-friendly.
PREVIEW_MAX_HEIGHT: int = 720

# --- Optional AI coach (Phase 9). The rest of the app works without these. --
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None


def ensure_directories() -> None:
    """Create the working folders if they do not exist yet."""
    for folder in (UPLOADS_DIR, OUTPUTS_DIR, DATA_DIR, ANALYSES_DIR, PREVIEWS_DIR,
                   ANNOTATED_DIR, TRACKING_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
