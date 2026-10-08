"""SwimVision - Streamlit entry point.

Run with:  streamlit run app.py

Page flow: with no active analysis we show the upload page; once the user
continues (or reopens a saved analysis from the sidebar) we show that
analysis' workspace (swimmer selection -> tracking results -> later phases).
"""

from __future__ import annotations

import logging

import streamlit as st

import config
from storage.analyses import cleanup_orphan_uploads, list_analyses, load_analysis
from ui.styles import inject_styles
from ui.upload import ACTIVE_ANALYSIS_KEY, render_upload_page
from ui.workspace import render_workspace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("swimvision")

st.set_page_config(page_title="SwimVision", page_icon="🏊", layout="wide")


def _render_header() -> None:
    st.markdown(
        '<div class="sv-hero"><h1>Swim<span class="sv-accent">Vision</span></h1>'
        "<p>Turn ordinary swim race footage into measurable performance data.</p></div>",
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    with st.sidebar:
        if st.button("＋ New analysis", width="stretch"):
            st.session_state.pop(ACTIVE_ANALYSIS_KEY, None)
            st.rerun()
        st.markdown("### Saved analyses")
        records = list_analyses()
        if not records:
            st.caption("None yet. Upload a video to create one.")
        for rec in records[:10]:
            status = "tracked" if rec.tracking else "not tracked yet"
            if st.button(rec.original_filename, key=f"open_{rec.analysis_id}", width="stretch",
                         help=f"{rec.race.label} · {rec.video.duration_s:.1f} s · id {rec.analysis_id}"):
                st.session_state[ACTIVE_ANALYSIS_KEY] = rec.analysis_id
                st.rerun()
            st.caption(f"{rec.race.label} · {status}")
        st.divider()
        st.caption("All video processing happens locally on this computer.")


def _render_main() -> None:
    active_id = st.session_state.get(ACTIVE_ANALYSIS_KEY)
    if active_id is None:
        render_upload_page()
        return
    try:
        record = load_analysis(active_id)
    except (FileNotFoundError, KeyError, ValueError):
        st.session_state.pop(ACTIVE_ANALYSIS_KEY, None)
        st.warning("That saved analysis could not be opened.")
        render_upload_page()
        return
    render_workspace(record)


def main() -> None:
    config.ensure_directories()
    cleanup_orphan_uploads()  # remove old unsaved uploads (privacy)
    inject_styles()
    _render_header()
    _render_sidebar()
    try:
        _render_main()
    except Exception:  # last-resort guard: never show a raw traceback to the user
        logger.exception("Unexpected error")
        st.error("Something unexpected went wrong. Please try again, or try a different video. "
                 "Details were written to the terminal log.")


main()
