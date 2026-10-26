"""FastAPI service for the SwimVision React application.

The service adapts the existing, tested tracking and race-analysis pipeline.
It deliberately does not contain tracking logic or change tracker settings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from core.coach import answer_coach_question
from core.errors import SwimVisionError
from core.geometry import BBox
from core.hybrid_tracking import load_hybrid_detector, track_swimmer
from core.models import RaceInfo
from core.person_detector import PersonDetector
from core.presentation import report_payload
from core.race_analysis import analyze_race
from core.sam2_tracking import Selection, results_table
from core.video import read_frame, read_video_metadata
from storage.analyses import (
    AnalysisRecord,
    load_analysis,
    new_analysis_id,
    save_analysis,
    save_tracking_table,
)

logger = logging.getLogger("swimvision.api")
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


class PointSelection(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class RunRequest(BaseModel):
    official_time_s: float | None = Field(default=None, gt=0)


class CoachRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


app = FastAPI(title="SwimVision API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
config.ensure_directories()

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()
_person_detector: PersonDetector | None = None
_person_detector_lock = Lock()


def parse_race_time(value: str | None) -> float | None:
    """Parse ``54.32`` or ``1:54.32`` into positive seconds."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 2:
                raise ValueError
            minutes, seconds = int(parts[0]), float(parts[1])
            if minutes < 0 or not 0 <= seconds < 60:
                raise ValueError
            result = minutes * 60 + seconds
        else:
            result = float(text)
        if result <= 0:
            raise ValueError
        return result
    except ValueError as exc:
        raise HTTPException(422, "Enter the official time as seconds (54.32) or minutes and seconds (1:54.32).") from exc


def _record(analysis_id: str) -> AnalysisRecord:
    try:
        return load_analysis(analysis_id)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(404, "That analysis could not be found.") from exc


def _set_job(analysis_id: str, **values: Any) -> None:
    with _jobs_lock:
        current = _jobs.setdefault(analysis_id, {})
        current.update(values)


def _job(analysis_id: str) -> dict[str, Any]:
    with _jobs_lock:
        return dict(_jobs.get(analysis_id, {}))


def _get_person_detector() -> PersonDetector:
    global _person_detector
    with _person_detector_lock:
        if _person_detector is None:
            _person_detector = PersonDetector(str(config.PERSON_MODEL_PATH), config.PERSON_CONFIDENCE)
        return _person_detector


def _box_from_tap(record: AnalysisRecord, point: PointSelection) -> BBox:
    """Choose the detected person under the tap, or a conservative fallback box."""
    frame = read_frame(record.absolute_video_path(), 0)
    px, py = point.x * record.video.width, point.y * record.video.height
    candidates: list[BBox] = []
    try:
        for item in _get_person_detector().detect(frame):
            candidates.append(BBox(item.x1, item.y1, item.x2, item.y2).clamp(record.video.width, record.video.height))
    except Exception:
        logger.exception("Opening-frame person detection failed; using the tapped location")

    containing = [box for box in candidates if box.x1 <= px <= box.x2 and box.y1 <= py <= box.y2]
    if containing:
        return min(containing, key=lambda box: box.area)

    if candidates:
        diagonal = max(1.0, (record.video.width ** 2 + record.video.height ** 2) ** 0.5)
        nearest = min(candidates, key=lambda box: (box.center[0] - px) ** 2 + (box.center[1] - py) ** 2)
        distance = ((nearest.center[0] - px) ** 2 + (nearest.center[1] - py) ** 2) ** 0.5
        if distance / diagonal <= 0.18:
            return nearest

    half_width = max(24.0, record.video.width * 0.055)
    half_height = max(36.0, record.video.height * 0.14)
    return BBox(px - half_width, py - half_height, px + half_width, py + half_height).clamp(
        record.video.width, record.video.height
    )


def _run_analysis(analysis_id: str, official_time_s: float | None) -> None:
    try:
        record = load_analysis(analysis_id)
        if not record.selections:
            raise SwimVisionError("Choose the swimmer before starting the analysis.")

        _set_job(analysis_id, state="running", stage="Following your swim", progress=0.08)
        annotated = config.ANNOTATED_DIR / f"{analysis_id}_tracked.mp4"
        detector = load_hybrid_detector()

        def tracking_progress(done: int, total: int) -> None:
            _set_job(analysis_id, progress=0.08 + 0.62 * (done / max(1, total)))

        frames, run = track_swimmer(
            record.absolute_video_path(), record.video, record.selections[0],
            record.video.frame_count - 1, annotated, analysis_width=960,
            progress=tracking_progress, detector=detector,
        )
        table = results_table(frames)
        record.tracking = run
        record.tracking_csv = save_tracking_table(analysis_id, table)
        record.annotated_video = annotated.relative_to(config.PROJECT_ROOT).as_posix()
        save_analysis(record)

        _set_job(analysis_id, stage="Reading your race", progress=0.72)

        def analysis_progress(done: int, total: int) -> None:
            _set_job(analysis_id, progress=0.72 + 0.21 * (done / max(1, total)))

        record.performance = analyze_race(
            record.absolute_video_path(), table, record.race, record.video,
            progress=analysis_progress, official_time_s=official_time_s,
        )
        _set_job(analysis_id, stage="Building your coaching plan", progress=0.96)
        save_analysis(record)
        _set_job(analysis_id, state="complete", progress=1.0)
    except Exception as exc:
        logger.exception("Analysis %s failed", analysis_id)
        message = str(exc) if isinstance(exc, SwimVisionError) else "The analysis could not finish. Please try the video again."
        _set_job(analysis_id, state="failed", message=message, progress=1.0)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyses", status_code=201)
async def create_analysis(
    video: UploadFile = File(...), stroke: str = Form(...), distance: int = Form(...),
    pool: str = Form(...), official_time: str | None = Form(default=None),
) -> dict[str, Any]:
    official_seconds = parse_race_time(official_time)
    try:
        race = RaceInfo(stroke=stroke, distance=distance, pool=pool)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    original_name = Path(video.filename or "race.mp4").name
    extension = Path(original_name).suffix.lower()
    if extension.lstrip(".") not in config.SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(415, "Choose an MP4, MOV, AVI, M4V, or MKV video.")
    analysis_id = new_analysis_id()
    path = config.UPLOADS_DIR / f"{analysis_id}{extension}"
    written = 0
    try:
        with path.open("wb") as target:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "This video is larger than the 2 GB upload limit.")
                target.write(chunk)
        metadata = read_video_metadata(path)
        record = AnalysisRecord(
            analysis_id=analysis_id, original_filename=original_name,
            video_path=path.relative_to(config.PROJECT_ROOT).as_posix(), race=race, video=metadata,
        )
        save_analysis(record)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except SwimVisionError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    finally:
        await video.close()
    return {"analysis_id": analysis_id, "official_time_s": official_seconds}


@app.get("/api/analyses/{analysis_id}/frame")
def opening_frame(analysis_id: str) -> Response:
    record = _record(analysis_id)
    frame = read_frame(record.absolute_video_path(), 0)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "The opening frame could not be prepared.")
    return Response(encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/analyses/{analysis_id}/selection")
def select_swimmer(analysis_id: str, point: PointSelection) -> dict[str, bool]:
    record = _record(analysis_id)
    record.selections = [Selection(0, _box_from_tap(record, point))]
    save_analysis(record)
    return {"selected": True}


@app.post("/api/analyses/{analysis_id}/run", status_code=202)
def run_analysis(analysis_id: str, request: RunRequest) -> dict[str, str]:
    record = _record(analysis_id)
    if not record.selections:
        raise HTTPException(409, "Tap the swimmer in the opening frame first.")
    current = _job(analysis_id)
    if current.get("state") == "running":
        return {"state": "running"}
    _set_job(analysis_id, state="running", stage="Preparing your race", progress=0.03)
    Thread(target=_run_analysis, args=(analysis_id, request.official_time_s), daemon=True).start()
    return {"state": "running"}


@app.get("/api/analyses/{analysis_id}/status")
def analysis_status(analysis_id: str) -> dict[str, Any]:
    record = _record(analysis_id)
    current = _job(analysis_id)
    if current:
        return current
    if record.performance is not None:
        return {"state": "complete", "stage": "Building your coaching plan", "progress": 1.0}
    return {"state": "waiting", "stage": "Preparing your race", "progress": 0.03}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict[str, Any]:
    record = _record(analysis_id)
    if record.performance is None:
        raise HTTPException(409, "This analysis is not finished yet.")
    return report_payload(record.performance)


@app.post("/api/analyses/{analysis_id}/coach")
def ask_coach(analysis_id: str, request: CoachRequest) -> dict[str, str]:
    record = _record(analysis_id)
    if record.performance is None:
        raise HTTPException(409, "Finish the analysis before asking the coach.")
    return {"answer": answer_coach_question(record.performance, request.question)}


# A production build can be served by the same process; Vite's dev server is
# still used during development so hot reload remains available.
_dist = config.PROJECT_ROOT / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
