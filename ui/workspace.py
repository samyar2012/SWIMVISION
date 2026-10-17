"""Workspace for one saved analysis: select swimmer -> track -> result."""

from __future__ import annotations

import streamlit as st

from storage.analyses import AnalysisRecord
from ui.selection import render_selection
from ui.tracking import render_tracking_controls, render_tracking_results


def render_workspace(record: AnalysisRecord) -> None:
    meta = record.video
    st.markdown(f"### {record.original_filename}")
    st.caption(f"{record.race.label} · {meta.resolution_label} · {meta.fps:.2f} fps · "
               f"{meta.frame_count:,} frames · {meta.duration_s:.2f} s")
    render_selection(record)
    st.divider()
    render_tracking_controls(record)
    st.divider()
    render_tracking_results(record)
