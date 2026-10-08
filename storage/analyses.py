"""Save and load analyses as small JSON files in ``data/analyses/``.

JSON was chosen over SQLite for now because each analysis is one self-contained
document that is easy to open, read and debug. Later phases will add keys such
as ``tracking``, ``strokes`` and ``events`` to the same record.

Privacy: only measurements and a *relative path* to the video are stored. The
video itself stays in the local ``uploads/`` folder.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config
from core.models import RaceInfo
from core.tracker import SwimmerSelection, TrackingSummary
from core.video import VideoMetadata


def new_analysis_id() -> str:
    """Short unique id used for filenames, e.g. ``'a3f9c2e1b7d4'``."""
    return uuid.uuid4().hex[:12]


@dataclass
class AnalysisRecord:
    """Everything we know about one analysed race."""

    analysis_id: str
    original_filename: str
    video_path: str  # relative to PROJECT_ROOT so the project folder is portable
    race: RaceInfo
    video: VideoMetadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # --- Phase 2: filled in after the user picks a swimmer and tracking runs ---
    selection: SwimmerSelection | None = None
    checkpoints: list[SwimmerSelection] = field(default_factory=list)  # all user boxes, incl. selection
    tracking: TrackingSummary | None = None
    tracking_csv: str | None = None        # relative path to the per-frame table
    annotated_video: str | None = None     # relative path to the overlay video

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "created_at": self.created_at,
            "original_filename": self.original_filename,
            "video_path": self.video_path,
            "race": self.race.to_dict(),
            "video": self.video.to_dict(),
            "selection": self.selection.to_dict() if self.selection else None,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "tracking": self.tracking.to_dict() if self.tracking else None,
            "tracking_csv": self.tracking_csv,
            "annotated_video": self.annotated_video,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisRecord":
        return cls(
            analysis_id=data["analysis_id"],
            original_filename=data["original_filename"],
            video_path=data["video_path"],
            race=RaceInfo.from_dict(data["race"]),
            video=VideoMetadata.from_dict(data["video"]),
            created_at=data["created_at"],
            # .get(): records saved by Phase 1 do not have these keys yet.
            selection=SwimmerSelection.from_dict(data["selection"]) if data.get("selection") else None,
            checkpoints=[SwimmerSelection.from_dict(c) for c in data.get("checkpoints", [])],
            tracking=TrackingSummary.from_dict(data["tracking"]) if data.get("tracking") else None,
            tracking_csv=data.get("tracking_csv"),
            annotated_video=data.get("annotated_video"),
        )

    def absolute_video_path(self) -> Path:
        return config.PROJECT_ROOT / self.video_path

    def absolute_annotated_path(self) -> Path | None:
        return config.PROJECT_ROOT / self.annotated_video if self.annotated_video else None


def _record_path(analysis_id: str, directory: Path | None = None) -> Path:
    return (directory or config.ANALYSES_DIR) / f"{analysis_id}.json"


def save_analysis(record: AnalysisRecord, directory: Path | None = None) -> Path:
    """Write ``record`` to disk and return the file path."""
    path = _record_path(record.analysis_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def load_analysis(analysis_id: str, directory: Path | None = None) -> AnalysisRecord:
    """Load one analysis by id. Raises ``FileNotFoundError`` if missing."""
    path = _record_path(analysis_id, directory)
    return AnalysisRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_tracking_table(analysis_id: str, table: pd.DataFrame) -> str:
    """Write the per-frame tracking table as CSV; return its path relative to the project."""
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TRACKING_DIR / f"{analysis_id}_tracking.csv"
    table.to_csv(path, index=False)
    return path.relative_to(config.PROJECT_ROOT).as_posix()


def load_tracking_table(record: AnalysisRecord) -> pd.DataFrame | None:
    """Load the saved tracking table for ``record`` (``None`` if there is none)."""
    if not record.tracking_csv:
        return None
    path = config.PROJECT_ROOT / record.tracking_csv
    return pd.read_csv(path) if path.is_file() else None


def cleanup_orphan_uploads(
    min_age_s: float = 3600,
    uploads_dir: Path | None = None,
    previews_dir: Path | None = None,
    analyses_dir: Path | None = None,
) -> list[Path]:
    """Delete uploads that no saved analysis uses (privacy + disk hygiene).

    Uploads become orphans when someone closes the browser before clicking
    "Create analysis session". Only files older than ``min_age_s`` are removed
    so an upload that is currently being looked at is never deleted.
    Returns the list of deleted files.
    """
    import time

    uploads = uploads_dir or config.UPLOADS_DIR
    previews = previews_dir or config.PREVIEWS_DIR
    used = {Path(r.video_path).name for r in list_analyses(analyses_dir)}
    used_stems = {Path(name).stem for name in used}
    deleted: list[Path] = []
    now = time.time()

    candidates = [p for p in uploads.glob("*") if p.is_file() and p.name != ".gitkeep"]
    candidates += [p for p in previews.glob("*_preview.mp4")] if previews.is_dir() else []
    for path in candidates:
        stem = path.stem.removesuffix("_preview")
        if stem in used_stems or now - path.stat().st_mtime < min_age_s:
            continue
        path.unlink(missing_ok=True)
        deleted.append(path)
    return deleted


def list_analyses(directory: Path | None = None) -> list[AnalysisRecord]:
    """Return all readable saved analyses, newest first. Corrupt files are skipped."""
    folder = directory or config.ANALYSES_DIR
    records: list[AnalysisRecord] = []
    for path in folder.glob("*.json"):
        try:
            records.append(AnalysisRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue  # a damaged file must not break the app
    return sorted(records, key=lambda r: r.created_at, reverse=True)
