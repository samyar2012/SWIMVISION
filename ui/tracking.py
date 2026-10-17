"""Steps 2 and 3: run SAM 2 tracking (synchronously) and report the result honestly."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

import config
from core.errors import SwimVisionError
from core.sam2_tracking import results_table, track_swimmer
from core.video import frame_to_timestamp, timestamp_to_frame
from storage.analyses import AnalysisRecord, load_tracking_table, save_analysis, save_tracking_table
from ui.styles import metric_card, step_label

logger = logging.getLogger(__name__)
QUICK_TEST_SECONDS = 5.0


def render_tracking_controls(record: AnalysisRecord) -> None:
    """Resolution + end time + "Run tracking" button."""
    step_label("Step 2 · Track the swimmer with SAM 2")
    marks, meta = record.selections, record.video
    if not marks:
        st.info("Mark your swimmer first (Step 1).")
        return
    last = meta.frame_count - 1
    first_mark, last_mark = marks[0].frame_index, marks[-1].frame_index
    if last_mark >= last:
        st.warning("The last mark is on the last frame of the video. Remove it or mark an earlier frame.")
        return

    start_s = frame_to_timestamp(first_mark, meta.fps)
    first_end_s = frame_to_timestamp(last_mark + 1, meta.fps)
    last_s = frame_to_timestamp(last, meta.fps)
    col_res, col_end = st.columns(2)
    res_label = col_res.radio("Analysis resolution", list(config.ANALYSIS_WIDTHS), horizontal=True,
                              key=f"res_{record.analysis_id}",
                              help="Frames are resized to this width before tracking. Results are always "
                                   "converted back to original video pixels.")
    end_s = col_end.number_input(
        "Stop at (seconds into the video)", min_value=round(first_end_s, 2), max_value=round(last_s, 2),
        value=round(min(max(start_s + QUICK_TEST_SECONDS, first_end_s + QUICK_TEST_SECONDS), last_s), 2),
        step=1.0, key=f"end_{record.analysis_id}_{first_mark}_{last_mark}",
    )
    end_frame = min(last, max(last_mark + 1, timestamp_to_frame(end_s, meta.fps)))
    st.caption(f"Tracking starts at your first mark (t = {start_s:.2f} s) and stops at "
               f"t = {frame_to_timestamp(end_frame, meta.fps):.2f} s. Try a short range first; "
               f"the full video ends at {last_s:.2f} s.")
    if st.button("Run tracking", type="primary"):
        _run_tracking(record, end_frame, config.ANALYSIS_WIDTHS[res_label])


def _run_tracking(record: AnalysisRecord, end_frame: int, analysis_width: int | None) -> None:
    st.caption("Tracking is running. Don't click anything until it finishes: "
               "any click restarts the page and cancels the run.")
    bar = st.progress(0.0, text="Loading SAM 2…")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Tracking frame {done} of {total}")

    annotated = config.ANNOTATED_DIR / f"{record.analysis_id}_sam2.mp4"
    try:
        results, run = track_swimmer(
            record.absolute_video_path(), record.video, record.selections, end_frame, analysis_width,
            annotated, debug_dir=config.DEBUG_DIR / record.analysis_id, progress=on_progress)
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
        st.warning("You changed the marks after this run. These results are for the previous marks; "
                   "run tracking again to update them.")

    t0, t1 = frame_to_timestamp(run.start_frame, fps), frame_to_timestamp(run.end_frame, fps)
    cards = [
        ("Analysed frames", f"{s.analyzed_frames}", f"t = {t0:.2f} – {t1:.2f} s"),
        ("Mask found", f"{s.tracked_fraction:.0%}", f"{s.tracked_frames} frames"),
        ("Lost frames", f"{s.lost_frames}", f"{len(s.lost_intervals)} interval(s) · "
                                            f"{s.conflict_frames} where passes disagree"),
        ("Longest loss", f"{s.longest_lost_s:.2f} s", f"{s.longest_lost_frames} frames"),
    ]
    for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
        col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)
    st.caption("“Mask found” means SAM 2 returned a mask on that frame. It does not prove the mask is on "
               "the right swimmer: watch the video. After a loss SAM 2 can pick up a different swimmer "
               "or a splash. Frames where the forward and backward passes found different things are "
               "counted as lost. Add a mark wherever a long loss starts and run again.")

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
        st.success("SAM 2 returned a mask on every analysed frame.")

    with st.expander("Run details"):
        st.write(f"Marks: {', '.join(f'frame {m.frame_index}' for m in run.selections)} · "
                 f"model: `{run.model}` · device: {run.device} · fp16: {run.half_precision}")
        st.write(f"Analysis size: {run.analysis_width}x{run.analysis_height} · every {run.stride} frame(s) · "
                 f"runtime {run.runtime_s:.1f} s")
        if run.debug_dir:
            st.write(f"Debug frames (mask drawn): `{run.debug_dir}`")
    table = load_tracking_table(record)
    if table is not None:
        st.download_button("Download per-frame tracking CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"{record.analysis_id}_tracking.csv", mime="text/csv")
