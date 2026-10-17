"""Swimmer selection (Step 4): mark the swimmer with one or more checkpoints, run tracking.

The app never assumes who the swimmer is. The user draws a box around their
swimmer on a frame - a *checkpoint*. More checkpoints (after the dive, after
each turn) let tracking recover wherever the swimmer was hidden underwater:
nothing in the picture shows where a submerged swimmer is, so the user points
them out again when they resurface.

Optionally, a person detector can propose numbered boxes, but on pool footage
those are mostly spectators (it was trained on people on land).
"""

from __future__ import annotations

import logging

import cv2
import streamlit as st

import config
from core.detector import PersonDetection, compute_device_name, detect_people, load_yolo
from core.errors import NoPersonDetectedError, SwimVisionError
from core.geometry import BBox, box_from_drag
from core.overlay import draw_manual_box, draw_person_candidates, zoomed_crop
from core.tracker import SwimmerSelection
from core.video import analysis_stride, frame_to_timestamp, read_frame
from storage.analyses import AnalysisRecord
from ui.jobs import start_job
from ui.styles import step_label

try:  # optional drawing component; the app falls back to sliders without it
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # pragma: no cover
    streamlit_image_coordinates = None

logger = logging.getLogger(__name__)

MANUAL_CHOICE = "manual"
METHOD_LABELS = {
    "sam2": "Segment Anything 2 - follows the swimmer's outline from your checkpoints (recommended)",
    "auto": "YOLO person detector + tracker (rarely sees swimmers in the water)",
    "visual": "Simple box tracker (fast, but loses the swimmer at the dive)",
}
CHECKPOINT_TIP = (
    "**Where to put checkpoints:** one on your swimmer on the block, one when they come up after the dive, "
    "and one after each turn. Underwater the swimmer cannot be seen by anyone - including the AI - so it "
    "needs you to point them out again when they resurface. Draw each box tightly around the swimmer."
)


@st.cache_resource(show_spinner="Loading detection model…")
def _cached_detector(model_name: str):
    """Keep the detection model in memory between Streamlit reruns."""
    return load_yolo(model_name)


def _render_settings() -> dict:
    """Advanced options with sensible defaults. Returns the chosen values."""
    with st.expander("Advanced tracking settings"):
        method = st.radio("Tracking method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
        if method != "sam2":
            st.caption("This method uses only your earliest checkpoint.")
        c1, c2 = st.columns(2)
        model = c1.selectbox("Person detector model", config.YOLO_MODEL_CHOICES,
                             index=config.YOLO_MODEL_CHOICES.index(config.DEFAULT_YOLO_MODEL),
                             help="Used by 'Find people in this frame' and the YOLO method.")
        conf = c2.slider("Minimum detection confidence", 0.05, 0.90, 0.25, 0.05,
                         help="Lower finds more (including false) detections.")
        imgsz = c1.selectbox("Detection image size", (640, 960, 1280), index=1,
                             help="Bigger helps find small/far people but is slower.")
        tracker = c2.selectbox("YOLO tracker", ("bytetrack.yaml", "botsort.yaml"),
                               help="BoT-SORT also compensates for camera movement.")
        st.caption(f"Compute device: {compute_device_name()}")
    return {"method": method, "model_name": model, "conf": conf, "imgsz": imgsz, "tracker_name": tracker}


def _last_sample_frame(frame_count: int, stride: int) -> int:
    return ((frame_count - 1) // stride) * stride


# ---------------------------------------------------------------------------
# Checkpoints live in st.session_state as {frame_index: {"box": [...], "source": ..., "confidence": ...}}
# ---------------------------------------------------------------------------
def _checkpoints(record: AnalysisRecord) -> dict[int, dict]:
    key = f"{record.analysis_id}_checkpoints"
    if key not in st.session_state:  # start from what was saved last time
        saved = record.checkpoints or ([record.selection] if record.selection else [])
        st.session_state[key] = {c.anchor_frame: {"box": c.bbox.to_list(), "source": c.source,
                                                  "confidence": c.confidence} for c in saved}
    return st.session_state[key]


def _as_selections(checkpoints: dict[int, dict]) -> list[SwimmerSelection]:
    return [SwimmerSelection(f, BBox.from_list(c["box"]), c["source"], c.get("confidence"))
            for f, c in sorted(checkpoints.items())]


def _render_checkpoint_list(record: AnalysisRecord, checkpoints: dict[int, dict], frame_key: str) -> None:
    """List the checkpoints with 'go to' and 'remove' buttons."""
    if not checkpoints:
        st.info("No checkpoints yet. Draw a box around your swimmer on the frame above.")
        return
    st.markdown(f"**Checkpoints ({len(checkpoints)})**")
    for f in sorted(checkpoints):
        c1, c2, c3 = st.columns([4, 1, 1])
        c1.write(f"t = {frame_to_timestamp(f, record.video.fps):.2f} s  (frame {f}) - "
                 f"{'drawn by you' if checkpoints[f]['source'] == 'manual' else 'detected person'}")
        c2.button("Go to", key=f"{frame_key}_goto_{f}", on_click=st.session_state.__setitem__,
                  args=(frame_key, f))
        if c3.button("Remove", key=f"{frame_key}_rm_{f}"):
            checkpoints.pop(f, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Optional person detection
# ---------------------------------------------------------------------------
def _detect_for_frame(video_path, frame_idx: int, settings: dict) -> list[PersonDetection]:
    frame = read_frame(video_path, frame_idx)
    detections = detect_people(_cached_detector(settings["model_name"]), frame,
                               settings["conf"], settings["imgsz"])
    if not detections:
        raise NoPersonDetectedError(
            "No people were detected on this frame. Draw a box around your swimmer instead.")
    return detections


def _choose_person(record: AnalysisRecord, frame_idx: int, settings: dict):
    """Returns ``(detections, choice)``; ``choice`` is a person number or ``MANUAL_CHOICE``."""
    key = f"{record.analysis_id}_det"
    det_key = (frame_idx, settings["model_name"], settings["conf"], settings["imgsz"])

    if st.button("Find people in this frame (optional)", key=f"{key}_btn"):
        try:
            with st.spinner("Detecting people…"):
                st.session_state[key] = {"key": det_key,
                                         "detections": _detect_for_frame(record.absolute_video_path(), frame_idx, settings)}
        except NoPersonDetectedError as exc:
            st.session_state[key] = {"key": det_key, "detections": []}
            st.warning(str(exc))
        except SwimVisionError as exc:
            st.error(str(exc))

    stored = st.session_state.get(key)
    detections: list[PersonDetection] = stored["detections"] if stored and stored["key"] == det_key else []
    if not detections:
        return [], MANUAL_CHOICE
    # Drawing the box yourself stays first: the person detector was trained on people on land
    # and, on pool footage, mostly finds spectators rather than swimmers.
    options: list = [MANUAL_CHOICE] + [d.number for d in detections]
    labels = {d.number: f"Person {d.number}  ({d.confidence:.0%})" for d in detections}
    labels[MANUAL_CHOICE] = "Draw the box myself (recommended)"
    st.info(f"{len(detections)} people detected. In pool footage these are often spectators or officials, "
            "not swimmers. If your swimmer has no box, draw one yourself.")
    choice = st.radio("How do you want to mark your swimmer on this frame?", options, index=0,
                      horizontal=True, format_func=labels.get, key=f"{key}_choice_{frame_idx}")
    return detections, choice


# ---------------------------------------------------------------------------
# Drawing a box
# ---------------------------------------------------------------------------
def _manual_box_sliders(frame_w: int, frame_h: int, key: str) -> BBox | None:
    """Fallback when the drawing component is unavailable: two range sliders."""
    st.caption("Drag the sliders until the box tightly surrounds your swimmer.")
    c1, c2 = st.columns(2)
    xr = c1.slider("Left ↔ right edge (pixels)", 0, frame_w, (int(frame_w * 0.35), int(frame_w * 0.65)), key=f"{key}_x")
    yr = c2.slider("Top ↕ bottom edge (pixels)", 0, frame_h, (int(frame_h * 0.35), int(frame_h * 0.65)), key=f"{key}_y")
    box = BBox(xr[0], yr[0], xr[1], yr[1])
    return box if box.width >= 8 and box.height >= 8 else None


def _draw_box_on_frame(frame, detections: list[PersonDetection], frame_idx: int,
                       checkpoints: dict[int, dict], key: str) -> None:
    """Let the user drag a rectangle on the frame; a new drag replaces this frame's checkpoint.

    ``streamlit-image-coordinates`` reports where the mouse was pressed and
    released (in displayed pixels); ``box_from_drag`` converts that to frame
    pixels. The component returns the same value on every rerun, so the last
    handled drag is remembered and only a *new* drag changes the checkpoint.
    """
    h, w = frame.shape[:2]
    existing = checkpoints.get(frame_idx)
    shown = draw_person_candidates(frame, detections) if detections else frame
    if existing:
        shown = draw_manual_box(shown, BBox.from_list(existing["box"]))
    st.caption("Press the mouse on one corner of your swimmer, drag to the opposite corner and release. "
               "Drag again to redraw.")
    drag = streamlit_image_coordinates(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), key=f"{key}_drag_{frame_idx}",
                                       click_and_drag=True, use_column_width="always", cursor="crosshair")
    seen_key = f"{key}_last_drag"
    if drag:
        stamp = (frame_idx, drag.get("x1"), drag.get("y1"), drag.get("x2"), drag.get("y2"))
        if st.session_state.get(seen_key) != stamp:  # a new drag, not a rerun of an old one
            st.session_state[seen_key] = stamp
            box = box_from_drag(drag, w, h)
            if box is None:
                st.warning("That box was too small. Press, drag across the swimmer and release.")
            else:
                checkpoints[frame_idx] = {"box": box.to_list(), "source": "manual", "confidence": None}
                st.rerun()  # redraw the image with the new box


def _render_marking(record: AnalysisRecord, frame, frame_idx: int, settings: dict,
                    checkpoints: dict[int, dict]) -> None:
    """Everything needed to set (or change) the checkpoint on the current frame."""
    detections, choice = _choose_person(record, frame_idx, settings)
    key = f"{record.analysis_id}_manual"
    if choice != MANUAL_CHOICE:
        picked = next((d for d in detections if d.number == choice), None)
        if picked is not None:
            checkpoints[frame_idx] = {"box": picked.bbox.to_list(), "source": "detected",
                                      "confidence": picked.confidence}
        st.image(cv2.cvtColor(draw_person_candidates(frame, detections, choice), cv2.COLOR_BGR2RGB),
                 width="stretch")
    elif streamlit_image_coordinates is not None:
        _draw_box_on_frame(frame, detections, frame_idx, checkpoints, key)
    else:  # component missing: slider fallback with a normal preview image
        box = _manual_box_sliders(frame.shape[1], frame.shape[0], key)
        if box is not None:
            st.image(cv2.cvtColor(draw_manual_box(frame, box), cv2.COLOR_BGR2RGB), width="stretch")
            if st.button("Set this box as the checkpoint for this frame", key=f"{key}_set"):
                checkpoints[frame_idx] = {"box": box.to_list(), "source": "manual", "confidence": None}

    current = checkpoints.get(frame_idx)
    if current:
        box = BBox.from_list(current["box"])
        c1, c2 = st.columns([3, 2])
        c1.image(cv2.cvtColor(zoomed_crop(frame, box), cv2.COLOR_BGR2RGB),
                 caption=f"Zoomed view of this checkpoint ({box.width:.0f} × {box.height:.0f} px) - "
                         "it should contain the whole swimmer and little else.")
        c2.success(f"Checkpoint set at t = {frame_to_timestamp(frame_idx, record.video.fps):.2f} s.")


def render_selection(record: AnalysisRecord) -> None:
    """The full selection UI. Runs tracking when the user confirms."""
    meta, path = record.video, record.absolute_video_path()
    if not path.is_file():
        st.error("The original video file is no longer in the uploads folder, so it cannot be analysed. "
                 "Please upload it again.")
        return
    stride = analysis_stride(meta.fps)
    step_label("Step 4 · Mark your swimmer")
    st.markdown(CHECKPOINT_TIP)

    checkpoints = _checkpoints(record)
    last = _last_sample_frame(meta.frame_count, stride)
    frame_key = f"{record.analysis_id}_frame"
    if frame_key not in st.session_state:
        st.session_state[frame_key] = min(checkpoints) if checkpoints else \
            min(last, int(round(meta.frame_count * 0.15 / stride)) * stride)
    frame_idx = st.slider("Frame", 0, last, step=stride, key=frame_key) if last > 0 else 0
    st.caption(f"t = {frame_to_timestamp(frame_idx, meta.fps):.2f} s  ·  analysis uses every {stride} frame(s) "
               f"- about {meta.fps / stride:.0f} analysed frames per second.")
    settings = _render_settings()

    try:
        frame = read_frame(path, frame_idx)
    except SwimVisionError as exc:
        st.error(str(exc))
        return
    _render_marking(record, frame, frame_idx, settings, checkpoints)
    _render_checkpoint_list(record, checkpoints, frame_key)

    if st.button("Analyze this swimmer", type="primary", disabled=not checkpoints,
                 key=f"{record.analysis_id}_go"):
        start_job(record, _as_selections(checkpoints), settings, stride)
        st.rerun()  # the workspace now shows the live progress view
