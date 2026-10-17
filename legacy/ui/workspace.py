"""Workspace for one saved analysis: video summary, swimmer selection, tracking results."""

from __future__ import annotations

import time

import streamlit as st

from core.tracker import MODE_SAM2
from storage.analyses import AnalysisRecord
from ui.jobs import TrackingJob, clear_job, get_job
from ui.selection import render_selection
from ui.styles import metric_card
from ui.tracking_results import render_tracking_results


def _render_summary(record: AnalysisRecord) -> None:
    v = record.video
    cards = [
        ("Race", record.race.label, record.original_filename),
        ("Frame rate", f"{v.fps:.3f} fps", f"{v.frame_count:,} frames"),
        ("Duration", f"{v.duration_s:.2f} s", "frames ÷ fps"),
        ("Resolution", v.resolution_label, f"codec: {v.codec}"),
    ]
    for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
        col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)


@st.fragment(run_every=1.0)
def _render_job_progress(job: TrackingJob) -> None:
    """Live progress, refreshed every second. Reloads the page once the job is done."""
    if job.done:
        st.rerun(scope="app")
    elapsed = int(time.time() - job.started_at)
    st.progress(job.fraction, text=job.message)
    st.caption(f"Running for {elapsed // 60} min {elapsed % 60:02d} s. This runs in the background: "
               "you can refresh the page or open another analysis - the work will not be lost.")


def render_workspace(record: AnalysisRecord) -> None:
    """Show the right steps for where this analysis is up to."""
    _render_summary(record)
    st.write("")
    if record.race.support_level != "supported":
        st.warning(f"{record.race.stroke} analysis is experimental. Tracking works for any stroke, "
                   "but stroke metrics (later phases) target freestyle first.")

    job = get_job(record.analysis_id)
    if job is not None and job.running:
        st.markdown("##### Analysing your swimmer…")
        _render_job_progress(job)
        return
    if job is not None and job.done:
        if job.error:
            st.error(job.error)
        clear_job(record.analysis_id)

    if record.tracking is None:
        render_selection(record)
        return
    old_method = record.tracking.mode != MODE_SAM2
    if old_method:
        st.warning("This result was made with an older tracking method that loses the swimmer at the dive. "
                   "Re-run it below: draw a box on your swimmer, add checkpoints where they come up after "
                   "the dive and after each turn, then press 'Analyze this swimmer'.")
    render_tracking_results(record)
    st.write("")
    needs_fix = old_method or record.tracking.followed_fraction < 0.5
    with st.expander("Re-run tracking (add checkpoints or choose a different swimmer)", expanded=needs_fix):
        render_selection(record)
