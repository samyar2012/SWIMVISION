"""Upload page: choose race details, upload a video, inspect its real metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

import config
from core.errors import SwimVisionError
from core.models import DISTANCES, POOLS, STROKE_SUPPORT, RaceInfo
from core.video import (
    VideoMetadata,
    frame_to_timestamp,
    is_browser_playable,
    make_browser_preview,
    read_frame,
    read_video_metadata,
)
from storage.analyses import AnalysisRecord, new_analysis_id, save_analysis
from ui.styles import metric_card, step_label

_STATE_KEY = "upload_state"
ACTIVE_ANALYSIS_KEY = "active_analysis_id"  # session key read by app.py


@dataclass
class UploadState:
    """What we remember between Streamlit reruns about the current upload."""

    file_id: str
    path: Path | None = None
    metadata: VideoMetadata | None = None
    error: str | None = None
    saved_analysis_id: str | None = None  # set once the user clicks "Create analysis"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stroke_label(stroke: str) -> str:
    level = STROKE_SUPPORT[stroke]
    return f"{stroke}  ·  fully supported" if level == "supported" else f"{stroke}  ·  experimental (coming later)"


def _discard_files(state: UploadState | None) -> None:
    """Delete the stored upload (and its preview) if it was never saved."""
    if state is None or state.saved_analysis_id or state.path is None:
        return
    state.path.unlink(missing_ok=True)
    (config.PREVIEWS_DIR / f"{state.path.stem}_preview.mp4").unlink(missing_ok=True)


def _process_new_upload(uploaded: Any) -> UploadState:
    """Write the upload to ``uploads/`` under a random name, then read its metadata.

    We never use the user's filename on disk (avoids path tricks and clashes).
    """
    state = UploadState(file_id=uploaded.file_id)
    ext = Path(uploaded.name).suffix.lower()
    dest = config.UPLOADS_DIR / f"{new_analysis_id()}{ext}"
    try:
        dest.write_bytes(uploaded.getvalue())
        state.path = dest
        state.metadata = read_video_metadata(dest)
    except SwimVisionError as exc:
        state.error = str(exc)
        dest.unlink(missing_ok=True)
        state.path = None
    except OSError as exc:  # disk full, permissions, ...
        state.error = f"Could not save the uploaded file: {exc.strerror or exc}"
        state.path = None
    return state


def _sync_state_with_uploader(uploaded: Any) -> UploadState | None:
    """Keep ``session_state`` in step with whatever the uploader currently holds."""
    current: UploadState | None = st.session_state.get(_STATE_KEY)
    if uploaded is None:
        # User pressed the "x" on the uploader: forget the file.
        _discard_files(current)
        st.session_state.pop(_STATE_KEY, None)
        return None
    if current is None or current.file_id != uploaded.file_id:
        _discard_files(current)
        with st.spinner("Reading video…"):
            current = _process_new_upload(uploaded)
        st.session_state[_STATE_KEY] = current
    return current


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def _render_race_form() -> RaceInfo:
    step_label("Step 1 · Race details")
    col_stroke, col_dist, col_pool = st.columns([2, 1, 2])
    stroke = col_stroke.selectbox("Stroke", list(STROKE_SUPPORT), format_func=_stroke_label)
    distance = col_dist.selectbox("Distance", DISTANCES, index=1)
    pool = col_pool.selectbox("Pool", list(POOLS), format_func=lambda p: POOLS[p][0])
    race = RaceInfo(stroke=stroke, distance=int(distance), pool=pool)
    if race.support_level != "supported":
        st.warning(
            f"{stroke} analysis is experimental and not implemented yet. "
            "SwimVision V1 is being built and validated for freestyle first."
        )
    return race


def _render_metadata_cards(meta: VideoMetadata) -> None:
    cards = [
        ("Frame rate", f"{meta.fps:.3f} fps", "read from the file"),
        ("Frames", f"{meta.frame_count:,}", "checked by decoding"),
        ("Duration", f"{meta.duration_s:.2f} s", "frames ÷ fps"),
        ("Resolution", meta.resolution_label, f"codec: {meta.codec}"),
    ]
    for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
        col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)


def _render_video_player(path: Path, meta: VideoMetadata) -> None:
    """Show the video, making a browser-friendly copy first if required."""
    if is_browser_playable(path, meta):
        st.video(str(path))
        return

    preview = config.PREVIEWS_DIR / f"{path.stem}_preview.mp4"
    try:
        if not preview.is_file():
            with st.spinner("Creating a browser-friendly preview (your original file is untouched)…"):
                make_browser_preview(path, preview)
        st.video(str(preview))
        st.caption("Preview re-encoded for playback. Analysis always uses your original file.")
    except SwimVisionError as exc:
        st.warning(f"Preview unavailable: {exc}")


def _render_frame_inspector(path: Path, meta: VideoMetadata) -> None:
    """Scrub to any frame and see its exact timestamp (proves FPS handling)."""
    st.markdown("##### Frame inspector")
    last = meta.frame_count - 1
    # st.slider requires min < max, so a single-frame video needs no slider.
    frame_idx = st.slider("Frame", 0, last, 0) if last > 0 else 0
    try:
        frame = read_frame(path, frame_idx)
    except SwimVisionError as exc:
        st.error(str(exc))
        return
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
    st.caption(f"Frame {frame_idx} of {meta.frame_count}  ·  t = {frame_to_timestamp(frame_idx, meta.fps):.3f} s")


def _render_create_analysis(state: UploadState, race: RaceInfo, original_name: str) -> None:
    assert state.path is not None and state.metadata is not None
    if state.saved_analysis_id:  # already saved (e.g. user came back to this page)
        st.session_state[ACTIVE_ANALYSIS_KEY] = state.saved_analysis_id
        st.rerun()
    if st.button("Continue to swimmer selection", type="primary"):
        record = AnalysisRecord(
            analysis_id=new_analysis_id(),
            original_filename=original_name,
            video_path=state.path.relative_to(config.PROJECT_ROOT).as_posix(),
            race=race,
            video=state.metadata,
        )
        save_analysis(record)
        state.saved_analysis_id = record.analysis_id
        st.session_state[ACTIVE_ANALYSIS_KEY] = record.analysis_id  # app.py now shows the workspace
        st.session_state.pop(_STATE_KEY, None)  # the upload is now owned by the saved analysis
        st.rerun()


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------
def render_upload_page() -> None:
    race = _render_race_form()

    st.write("")
    step_label("Step 2 · Upload race video")
    uploaded = st.file_uploader(
        "Upload a side-view freestyle race video",
        type=list(config.SUPPORTED_VIDEO_EXTENSIONS),
        help="Videos are processed locally on this computer and are not sent anywhere.",
    )
    state = _sync_state_with_uploader(uploaded)
    if state is None:
        st.info("Choose a video to begin. Best results: one clearly visible swimmer, "
                "camera at the side of the pool, fairly steady.")
        return
    if state.error:
        st.error(state.error)
        return

    assert state.path is not None and state.metadata is not None
    st.write("")
    step_label("Step 3 · Video check")
    _render_metadata_cards(state.metadata)
    st.write("")
    _render_video_player(state.path, state.metadata)
    _render_frame_inspector(state.path, state.metadata)
    st.write("")
    _render_create_analysis(state, race, uploaded.name)
