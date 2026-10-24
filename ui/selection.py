"""Step 1: identify the swimmer once on the opening video frame.

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
from core.video import read_frame
from storage.analyses import AnalysisRecord, save_analysis
from ui.styles import step_label


def _rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _starting_box(record: AnalysisRecord, frame_idx: int) -> BBox:
    """Where the rectangle starts: this frame's mark, else the nearest mark's box, else a small centred box."""
    if record.selections:
        return min(record.selections, key=lambda m: abs(m.frame_index - frame_idx)).box
    w, h = record.video.width, record.video.height
    return BBox(w * 0.45, h * 0.40, w * 0.55, h * 0.60)


def _set_mark(record: AnalysisRecord, mark: Selection) -> None:
    """Keep the one visual selection used to initialize persistent tracking."""
    record.selections = [mark]
    record.tracking = None
    record.tracking_csv = None
    record.annotated_video = None
    save_analysis(record)


def render_selection(record: AnalysisRecord) -> None:
    """One rectangle on the opening frame; no time or interval input."""
    meta = record.video
    step_label("Step 1 · Choose your swimmer")
    st.markdown("Drag the yellow box around your swimmer in the opening frame. "
                "That is the only selection—there is no start time, end time, or second mark.")

    frame_idx = 0
    size_label = st.radio(
        "Image size", list(config.SELECTION_DISPLAY_WIDTHS), horizontal=True,
        key=f"sel_size_{record.analysis_id}",
    )
    st.caption("Opening frame · tracking will cover the entire video")

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
        col_zoom, col_button = st.columns([3, 2])
        col_zoom.image(_rgb(zoomed_crop(frame, box)), caption="Your swimmer")
        label = "Update chosen swimmer" if record.selections else "Use this swimmer"
        if col_button.button(label, type="primary", width="stretch"):
            _set_mark(record, Selection(frame_idx, box))
            st.rerun()
    _render_marks(record)


def _render_marks(record: AnalysisRecord) -> None:
    """Confirm that the app saved the one visual swimmer selection."""
    if not record.selections:
        st.info("Draw the yellow box around your swimmer, then click **Use this swimmer**.")
        return
    mark = record.selections[0]
    st.success("Swimmer chosen. No times, frame numbers, coordinates, or lane boundaries are needed.")
