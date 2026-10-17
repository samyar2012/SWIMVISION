"""Tracking results (Step 5): honest summary, annotated video, position graph.

Everything shown here is read from the saved tracking table / summary. Pixel
positions are labelled as pixels: without calibration (Phase 7) they are NOT
metres or yards.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.tracker import (
    MODE_SAM2,
    MODE_VISUAL,
    STATUS_LOST,
    STATUS_TEXT,
    TrackingSummary,
    UnreliableSegment,
)
from storage.analyses import AnalysisRecord, load_tracking_table
from ui.styles import metric_card, step_label

MODE_TEXT = {"yolo_tracker": "YOLO + tracker", MODE_VISUAL: "Visual box tracker",
             MODE_SAM2: "Segment Anything 2"}


def _segments_table(segments: list[UnreliableSegment]) -> pd.DataFrame:
    return pd.DataFrame([{
        "From (s)": round(s.start_time_s, 2),
        "To (s)": round(s.end_time_s, 2),
        "What happened": "; ".join(STATUS_TEXT.get(r, r) for r in s.reasons),
    } for s in segments])


def _render_cards(summary: TrackingSummary) -> None:
    n = max(summary.n_samples, 1)
    check = summary.n_reacquired + summary.n_not_moving + summary.n_low_confidence
    cards = [
        ("Swimmer followed", f"{summary.followed_fraction:.0%}", "of the video, with no warning"),
        ("Swimmer lost", f"{summary.n_lost / n:.0%}", "underwater, in splash, too small, or tracker lost them"),
        ("Check these", f"{check / n:.0%}", "found again after a gap, or box not moving"),
        ("Method", MODE_TEXT.get(summary.mode, summary.mode), f"processed in {summary.runtime_s:.0f} s"),
    ]
    for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
        col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)


def _render_warnings(summary: TrackingSummary) -> None:
    """Never hide tracking problems - and always say what they mean."""
    if summary.followed_fraction < 0.5:
        st.error(f"Your swimmer was followed without problems in only {summary.followed_fraction:.0%} of the "
                 "video, so measurements from this tracking would not be reliable. The notes below say where "
                 "it went wrong. Fix it by adding checkpoints there (open 'Re-run tracking' further down).")
    elif summary.segments:
        st.warning("Some stretches need checking (listed below). Watch them in the video before trusting "
                   "measurements from those times.")
    for note in summary.notes:
        st.info(note)


def _position_figure(table: pd.DataFrame, segments: list[UnreliableSegment]) -> go.Figure:
    fig = go.Figure()
    # connectgaps=False: lost frames have no coordinates, so the line visibly breaks there.
    fig.add_trace(go.Scatter(x=table["timestamp_s"], y=table["cx"], name="Horizontal position (x)",
                             mode="lines", line=dict(color="#22D3EE", width=2), connectgaps=False))
    fig.add_trace(go.Scatter(x=table["timestamp_s"], y=table["cy"], name="Vertical position (y)",
                             mode="lines", line=dict(color="#A78BFA", width=2), connectgaps=False))
    for seg in segments:
        fig.add_vrect(x0=seg.start_time_s, x1=seg.end_time_s, fillcolor="#EF4444", opacity=0.18,
                      line_width=0, annotation_text="check" if STATUS_LOST not in seg.reasons else "lost",
                      annotation_position="top left")
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Time (s)", yaxis_title="Position in frame (pixels)",
                      legend=dict(orientation="h", y=1.12))
    return fig


def render_tracking_results(record: AnalysisRecord) -> None:
    summary = record.tracking
    if summary is None:
        return
    step_label("Step 5 · Tracking result")
    _render_cards(summary)
    st.write("")
    _render_warnings(summary)

    annotated = record.absolute_annotated_path()
    if annotated is not None and annotated.is_file():
        st.video(str(annotated))
        st.caption("Analysed video: the box on your swimmer, their recent path, and a banner whenever the "
                   "swimmer is not visible or something needs checking.")
    else:
        st.warning("The annotated video file is missing. Run the analysis again to recreate it.")

    table = load_tracking_table(record)
    if table is None:
        st.warning("The saved tracking table is missing. Run the analysis again to recreate it.")
        return

    st.markdown("##### Swimmer position over time")
    st.plotly_chart(_position_figure(table, summary.segments), width="stretch")
    st.caption("Positions are in image pixels, not metres/yards - distance needs pool calibration "
               "(coming in a later phase). Red bands mark segments to check; gaps are lost frames.")

    if summary.segments:
        st.markdown("##### Segments to check")
        st.dataframe(_segments_table(summary.segments), hide_index=True, width="stretch")
    st.download_button("Download tracking data (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"swimvision_{record.analysis_id}_tracking.csv", mime="text/csv")
