"""Steps 2 and 3: run automatic hybrid tracking and show the result."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

import config
from core.errors import SwimVisionError
from core.hybrid_tracking import load_hybrid_detector, track_swimmer
from core.sam2_tracking import results_table
from core.video import frame_to_timestamp
from storage.analyses import AnalysisRecord, load_tracking_table, save_analysis, save_tracking_table
from ui.styles import metric_card, step_label

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _detector(
    person_model_path: str,
    swimmer_model_path: str,
    person_confidence: float,
    swimmer_threshold: float,
    merge_iou_threshold: float,
):
    """Cache one detector for one exact model/configuration combination.

    The arguments intentionally form Streamlit's cache key. Without them,
    changing a threshold or model could keep an old detector object alive
    across an app reload even though the source code had changed.
    """
    return load_hybrid_detector()


def render_tracking_controls(record: AnalysisRecord) -> None:
    """One-button persistent tracking from the visual selection to video end."""
    step_label("Step 2 · Track your swimmer")
    marks, meta = record.selections, record.video
    if not marks:
        st.info("Choose your swimmer first (Step 1).")
        return
    last = meta.frame_count - 1
    first_mark = marks[0].frame_index
    if first_mark >= last:
        st.warning("Choose your swimmer earlier in the video so there is something to track.")
        return

    start_s = frame_to_timestamp(first_mark, meta.fps)
    last_s = frame_to_timestamp(last, meta.fps)
    st.write(f"SwimVision will follow the selected swimmer from {start_s:.2f} seconds to the end "
             f"of the video ({last_s:.2f} seconds), including the dive and underwater gaps.")
    if st.button("Track my swimmer", type="primary", width="stretch"):
        _run_tracking(record, last)


def _run_tracking(record: AnalysisRecord, end_frame: int) -> None:
    st.caption("Tracking is running. You can leave this tab open until it finishes.")
    bar = st.progress(0.0, text="Loading swimmer models…")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Tracking frame {done} of {total}")

    annotated = config.ANNOTATED_DIR / f"{record.analysis_id}_hybrid.mp4"
    try:
        detector = _detector(
            str(config.PERSON_MODEL_PATH),
            str(config.SWIMMER_MODEL_PATH),
            config.PERSON_CONFIDENCE,
            config.SWIMMER_THRESHOLD,
            config.MERGE_IOU_THRESHOLD,
        )
        results, run = track_swimmer(
            record.absolute_video_path(), record.video, record.selections[0], end_frame, annotated,
            analysis_width=960, progress=on_progress, detector=detector)
    except SwimVisionError as exc:
        logger.exception("Tracking failed")
        bar.empty()
        st.error(str(exc))
        if config.DEBUG:
            st.exception(exc)
        return
    record.tracking = run
    record.tracking_csv = save_tracking_table(record.analysis_id, results_table(results))
    record.annotated_video = annotated.relative_to(config.PROJECT_ROOT).as_posix()
    save_analysis(record)
    st.rerun()


def render_tracking_results(record: AnalysisRecord) -> None:
    """Counts, lost intervals, the annotated video and the per-frame CSV."""
    run = record.tracking
    if run is None:
        return
    s, fps = run.stats, record.video.fps
    step_label("Step 3 · Tracking result")
    if tuple(record.selections) != run.selections:
        st.warning("You changed the chosen swimmer after this run. These results are for the previous selection; "
                   "run tracking again to update them.")

    t0, t1 = frame_to_timestamp(run.start_frame, fps), frame_to_timestamp(run.end_frame, fps)
    cards = [
        ("Analysed frames", f"{s.analyzed_frames}", f"t = {t0:.2f} – {t1:.2f} s"),
        ("Swimmer followed", f"{s.tracked_fraction:.0%}", f"{s.tracked_frames} frames"),
        ("Lost frames", f"{s.lost_frames}", f"{len(s.lost_intervals)} interval(s) · "
                                            "where the tracker was not confident"),
        ("Longest loss", f"{s.longest_lost_s:.2f} s", f"{s.longest_lost_frames} frames"),
    ]
    for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
        col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)
    st.caption("Green boxes are detector observations. Yellow boxes are short motion predictions through "
               "a splash or underwater gap. Watch the result once to confirm it stayed on your swimmer.")

    video = record.absolute_annotated_path()
    if video and video.is_file():
        st.video(str(video))
        out = run.output_video
        st.caption(f"Annotated video checked after writing: {out['frames']} frames, {out['fps']} fps, "
                   f"{out['duration_s']} s, {out['width']}x{out['height']}, codec {out['codec']}.")
    else:
        st.error("The annotated video file is missing. Run tracking again.")

    st.markdown("##### Frames where the swimmer was lost")
    if s.lost_intervals:
        st.dataframe(pd.DataFrame([{
            "from (s)": round(i.start_s, 2), "to (s)": round(i.end_s, 2), "frames": i.n_frames,
            "duration (s)": round(i.duration_s, 2), "first frame": i.start_frame, "last frame": i.end_frame,
        } for i in s.lost_intervals]), hide_index=True, width="stretch")
    else:
        st.success("The tracker followed the swimmer across every analysed frame.")

    with st.expander("Run details"):
        st.write(f"Selection time: {frame_to_timestamp(run.selections[0].frame_index, fps):.2f} s · "
                 f"model: `{run.model}` · device: {run.device}")
        st.write(f"Analysis size: {run.analysis_width}x{run.analysis_height} · every {run.stride} frame(s) · "
                 f"runtime {run.runtime_s:.1f} s")
    table = load_tracking_table(record)
    if table is not None:
        st.download_button("Download per-frame tracking CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"{record.analysis_id}_tracking.csv", mime="text/csv")
