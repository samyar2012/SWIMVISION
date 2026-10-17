"""Step 1: mark the swimmer - drag a rectangle around them on one or more frames.

Uses ``streamlit-cropper``: a canvas with a visible rectangle the user can move
and resize. We resize the frame ourselves (``should_resize_image=False``) so
the display size is known exactly, and convert the rectangle back to original
video pixels with ``core.geometry.cropper_rect_to_box``.

After the user moves the rectangle the component reports fabric.js'
``getBoundingRect()``, which includes the outline width; a 1 px outline keeps
that error at <= 1 display pixel.
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

import config
from core.errors import SwimVisionError
from core.geometry import BBox, box_to_cropper_coords, cropper_rect_to_box, display_size
from core.overlay import zoomed_crop
from core.sam2_tracking import Selection
from core.video import frame_to_timestamp, read_frame
from storage.analyses import AnalysisRecord, save_analysis
from ui.styles import step_label


def _rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _box_text(box: BBox) -> str:
    x1, y1, x2, y2 = box.to_int_list()
    return f"x1 = {x1}  y1 = {y1}  x2 = {x2}  y2 = {y2}"


def _starting_box(record: AnalysisRecord, frame_idx: int) -> BBox:
    """Where the rectangle starts: this frame's mark, else the nearest mark's box, else a small centred box."""
    if record.selections:
        return min(record.selections, key=lambda m: abs(m.frame_index - frame_idx)).box
    w, h = record.video.width, record.video.height
    return BBox(w * 0.45, h * 0.40, w * 0.55, h * 0.60)


def _set_mark(record: AnalysisRecord, mark: Selection) -> None:
    """Add ``mark``, replacing any mark already on the same frame."""
    others = [m for m in record.selections if m.frame_index != mark.frame_index]
    record.selections = sorted([*others, mark], key=lambda m: m.frame_index)
    save_analysis(record)


def render_selection(record: AnalysisRecord) -> None:
    """Frame slider + rectangle tool + zoomed check + the list of marks."""
    meta = record.video
    step_label("Step 1 · Mark your swimmer")
    st.markdown(
        "1. Scrub to a frame where your swimmer is clearly visible (e.g. on the block), draw the box, "
        "click **Mark swimmer on this frame**.\n"
        "2. **Scrub to just after the dive, when the swimmer first shows in the water, and mark them "
        "again.** Without this second mark SAM 2 loses the swimmer at water entry; with it, tracking "
        "runs backward from that mark through the dive and forward from it into the race.\n"
        "3. Add more marks wherever the swimmer is lost later (e.g. after a turn).")

    last = meta.frame_count - 1
    start = record.selections[0].frame_index if record.selections else 0
    col_slider, col_size = st.columns([4, 1])
    frame_idx = col_slider.slider("Frame", 0, max(last, 1), min(start, last),
                                  key=f"sel_frame_{record.analysis_id}") if last > 0 else 0
    size_label = col_size.radio("Image size", list(config.SELECTION_DISPLAY_WIDTHS),
                                key=f"sel_size_{record.analysis_id}")
    st.caption(f"Frame {frame_idx} · t = {frame_to_timestamp(frame_idx, meta.fps):.3f} s · "
               "use the arrow keys on the slider for single frames")

    try:
        frame = read_frame(record.absolute_video_path(), frame_idx)
    except SwimVisionError as exc:
        st.error(str(exc))
        return

    dw, dh = display_size(meta.width, meta.height, config.SELECTION_DISPLAY_WIDTHS[size_label])
    interp = cv2.INTER_AREA if dw < meta.width else cv2.INTER_CUBIC
    shown = Image.fromarray(_rgb(cv2.resize(frame, (dw, dh), interpolation=interp)))
    st.markdown("**Drag the yellow rectangle onto your swimmer; drag its corners to resize it.**")
    rect = st_cropper(
        shown, realtime_update=True, box_color="#FFC800", return_type="box", should_resize_image=False,
        default_coords=box_to_cropper_coords(_starting_box(record, frame_idx), dw, dh, meta.width, meta.height),
        key=f"cropper_{record.analysis_id}_{frame_idx}_{dw}", stroke_width=1,
    )
    box = cropper_rect_to_box(rect, dw, dh, meta.width, meta.height)
    if box is None:
        st.warning("The rectangle is too small. Make it cover the whole swimmer.")
    else:
        st.code(f"Selected box: {_box_text(box)}   (original video pixels, frame {frame_idx}, "
                f"video {meta.width}x{meta.height})", language=None)
        col_zoom, col_button = st.columns([3, 2])
        col_zoom.image(_rgb(zoomed_crop(frame, box)), caption="Zoomed check: is the whole swimmer inside the box?")
        on_frame = any(m.frame_index == frame_idx for m in record.selections)
        label = "Update the mark on this frame" if on_frame else "Mark swimmer on this frame"
        if col_button.button(label, type="primary", width="stretch"):
            _set_mark(record, Selection(frame_idx, box))
            st.rerun()
    _render_marks(record)


def _render_marks(record: AnalysisRecord) -> None:
    """List every mark with a zoomed check and a remove button."""
    if not record.selections:
        st.info("No marks yet. Draw the box and click **Mark swimmer on this frame**.")
        return
    st.markdown(f"##### Marks ({len(record.selections)}) - tracking uses these")
    cols = st.columns(min(len(record.selections), 4))
    for n, mark in enumerate(record.selections):
        with cols[n % len(cols)]:
            t = frame_to_timestamp(mark.frame_index, record.video.fps)
            try:
                crop = zoomed_crop(read_frame(record.absolute_video_path(), mark.frame_index), mark.box,
                                   target_height=160, max_width=320)
                st.image(_rgb(crop))
            except SwimVisionError as exc:
                st.error(str(exc))
            st.caption(f"Mark {n + 1} · frame {mark.frame_index} · t = {t:.2f} s\n\n{_box_text(mark.box)}")
            if st.button("Remove", key=f"rm_mark_{record.analysis_id}_{mark.frame_index}"):
                record.selections = [m for m in record.selections if m.frame_index != mark.frame_index]
                save_analysis(record)
                st.rerun()
    if len(record.selections) == 1:
        st.warning("Only one mark. If the race starts with a dive, add a second mark just after the "
                   "swimmer surfaces, or tracking will most likely stop at water entry.")
