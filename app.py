"""SwimVision - Streamlit entry point.

Run with:  streamlit run app.py
Developer mode (full tracebacks in the page):  set SWIMVISION_DEBUG=1 first.

Page flow: with no active analysis we show the upload page; once the user
continues (or reopens a saved analysis from the sidebar) we show that
analysis' workspace (visually select one swimmer -> automatic tracking -> result).
"""

from __future__ import annotations

import logging

import streamlit as st

import config
from storage.analyses import cleanup_orphan_uploads, list_analyses, load_analysis
from ui.styles import inject_styles
from ui.upload import ACTIVE_ANALYSIS_KEY, render_upload_page
from ui.workspace import render_workspace

logger = logging.getLogger("swimvision")

st.set_page_config(page_title="SwimVision", page_icon="🏊", layout="wide")


@st.cache_resource
def _configure_logging() -> None:
    """Log to the terminal; in developer mode also to outputs/logs/swimvision.log (once per process)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.DEBUG:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.LOGS_DIR / "swimvision.log", encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if config.DEBUG else logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def _render_header() -> None:
    st.markdown(
        '<div class="sv-hero"><h1>Swim<span class="sv-accent">Vision</span></h1>'
        "<p>Turn ordinary swim race footage into measurable performance data.</p></div>",
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    with st.sidebar:
        if config.DEBUG:
            st.warning("Developer mode (SWIMVISION_DEBUG=1): full errors are shown.")
        if st.button("＋ New analysis", width="stretch"):
            st.session_state.pop(ACTIVE_ANALYSIS_KEY, None)
            st.rerun()
        st.markdown("### Saved analyses")
        records = list_analyses()
        if not records:
            st.caption("None yet. Upload a video to create one.")
        for rec in records[:10]:
            status = f"verified {rec.tracking.stats.tracked_fraction:.0%}" if rec.tracking else "not tracked yet"
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
    except FileNotFoundError:
        st.session_state.pop(ACTIVE_ANALYSIS_KEY, None)
        st.warning("That saved analysis no longer exists.")
        render_upload_page()
        return
    render_workspace(record)


def main() -> None:
    _configure_logging()
    config.ensure_directories()
    cleanup_orphan_uploads()  # remove old unsaved uploads (privacy)
    inject_styles()
    _render_header()
    _render_sidebar()
    try:
        _render_main()
    except Exception as exc:  # Streamlit's own rerun/stop signals are BaseException and pass through
        logger.exception("Unexpected error")
        if config.DEBUG:
            st.error(f"{type(exc).__name__}: {exc}")
            st.exception(exc)
        else:
            st.error(f"Something went wrong: {type(exc).__name__}: {exc}\n\n"
                     "The full traceback is in the terminal. Start the app with SWIMVISION_DEBUG=1 "
                     "to see it here.")


main()
