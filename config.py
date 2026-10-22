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

# Developer mode: SWIMVISION_DEBUG=1 shows full tracebacks in the app and
# writes them to outputs/logs/swimvision.log.
DEBUG: bool = os.getenv("SWIMVISION_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

# --- Folders (all git-ignored except for .gitkeep placeholders) -------------
UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"      # raw user videos (private!)
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"      # annotated videos, previews, debug frames
DATA_DIR: Path = PROJECT_ROOT / "data"            # saved analyses (JSON)
ANALYSES_DIR: Path = DATA_DIR / "analyses"
PREVIEWS_DIR: Path = OUTPUTS_DIR / "previews"     # browser-friendly video copies
ANNOTATED_DIR: Path = OUTPUTS_DIR / "annotated"   # videos with the tracking overlay
DEBUG_DIR: Path = OUTPUTS_DIR / "debug"           # sample frames with the mask drawn
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
TRACKING_DIR: Path = DATA_DIR / "tracking"        # per-frame tracking tables (CSV)
MODELS_DIR: Path = PROJECT_ROOT / "models"        # downloaded pretrained weights

# --- Hybrid persistent tracker ------------------------------------------------
PERSON_MODEL_PATH: Path = PROJECT_ROOT / "yolo26n.pt"  # Ultralytics downloads it on first use
SWIMMER_MODEL_PATH: Path = MODELS_DIR / "colorShifts_deeper_zoomedOut_200epochs.pth"
PERSON_CONFIDENCE: float = 0.25
# Match the standalone heatmap test exactly. Lowering this threshold causes
# nearby weak activations to join during morphology and can merge lanes.
SWIMMER_THRESHOLD: float = 0.35
MERGE_IOU_THRESHOLD: float = 0.15

# --- Tracking (Phase 2): SAM 2 video segmentation, one swimmer ---------------
SAM2_MODEL_PATH: Path = MODELS_DIR / "sam2.1_b.pt"  # Segment Anything 2.1 base, ~155 MB, auto-downloaded
SAM2_IMGSZ: int = 1024                              # SAM 2's native input size
TARGET_ANALYSIS_FPS: float = 30.0                   # 60 fps videos are analysed every 2nd frame
# Width the frames are resized to before tracking (None = original). Results
# are always converted back to original video pixels.
ANALYSIS_WIDTHS: dict[str, int | None] = {"640 px (fastest)": 640, "960 px": 960, "Original": None}
DEBUG_FRAME_EVERY: int = 50                         # save debug/frame_XXX.jpg every N analysed frames

# --- Swimmer selection screen -----------------------------------------------
SELECTION_DISPLAY_WIDTHS: dict[str, int] = {"Medium": 900, "Large": 1200, "Small": 700}

# --- Video handling ---------------------------------------------------------
# Containers we accept in the uploader. Whether a *specific* file can be
# decoded is checked separately by OpenCV after upload.
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = ("mp4", "mov", "avi", "m4v", "mkv")

# Longest preview side we re-encode to when the original is not browser-friendly.
PREVIEW_MAX_HEIGHT: int = 720

# --- Optional AI coach (later phase). The rest of the app works without these.
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None


def ensure_directories() -> None:
    """Create the working folders if they do not exist yet."""
    for folder in (UPLOADS_DIR, OUTPUTS_DIR, DATA_DIR, ANALYSES_DIR, PREVIEWS_DIR,
                   ANNOTATED_DIR, DEBUG_DIR, LOGS_DIR, TRACKING_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
