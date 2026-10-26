"""Step 4: run and explain the race analysis in plain language."""

from __future__ import annotations

import json
import logging

import pandas as pd
import streamlit as st

from core.coach import CoachUnavailable, ask_coach
from core.errors import SwimVisionError
from core.race_analysis import RaceAnalysis, analyze_race
from core.race_report import format_time, length_label, strokes_word
from storage.analyses import AnalysisRecord, load_tracking_table, save_analysis
from ui.styles import metric_card, step_label

logger = logging.getLogger(__name__)


def render_analysis(record: AnalysisRecord) -> None:
    step_label("Step 4 · Understand your race")
    st.write("Your splits, how the race went, and what to work on next.")
    official_time_s = _official_time_input(record)
    if record.performance is None:
        if st.button("Analyze my race", type="primary", width="stretch"):
            if official_time_s is not False:
                _run_analysis(record, official_time_s)
        return
    _render_report(record)
    if st.button("Update my analysis", width="stretch"):
        if official_time_s is not False:
            _run_analysis(record, official_time_s)


def _parse_race_time(value: str) -> float:
    """Accept familiar stopwatch formats: SS.ss or M:SS.ss."""
    text = value.strip()
    if not text:
        raise ValueError("Enter your race time.")
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 2:
                raise ValueError
            minutes, seconds = int(parts[0]), float(parts[1])
            if minutes < 0 or not 0 <= seconds < 60:
                raise ValueError
            total = minutes * 60 + seconds
        else:
            total = float(text)
    except ValueError as exc:
        raise ValueError("Use seconds (54.32) or minutes and seconds (1:54.32).") from exc
    if not 0 < total < 3600:
        raise ValueError("Race time must be greater than zero and less than one hour.")
    return round(total, 3)


def _official_time_input(record: AnalysisRecord) -> float | None | bool:
    current = record.official_time_s
    with st.expander("Add my official race time (optional)", expanded=current is not None):
        st.caption("Your official time makes the total and the last split exact.")
        use_official = st.checkbox(
            "Use a time from the scoreboard or stopwatch",
            value=current is not None,
            key=f"official-time-enabled-{record.analysis_id}",
        )
        value = st.text_input(
            "Official race time",
            value=format_time(current) if current is not None else "",
            placeholder="Example: 54.32 or 1:54.32",
            disabled=not use_official,
            key=f"official-time-value-{record.analysis_id}",
        )
        if not use_official:
            return None
        try:
            return _parse_race_time(value)
        except ValueError as exc:
            st.warning(str(exc))
            return False


def _run_analysis(record: AnalysisRecord, official_time_s: float | None = None) -> None:
    table = load_tracking_table(record)
    if table is None:
        st.error("The tracking measurements are missing. Run tracking again first.")
        return
    bar = st.progress(0.0, text="Timing your race…")

    def progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Timing your race… {done / total:.0%}")

    try:
        record.performance = analyze_race(
            record.absolute_video_path(), table, record.race, record.video, progress,
            official_time_s=official_time_s,
        )
    except SwimVisionError as exc:
        logger.exception("Race analysis failed")
        bar.empty()
        st.error(str(exc))
        return
    record.official_time_s = official_time_s
    save_analysis(record)
    st.rerun()


def plain_report(report: RaceAnalysis) -> str:
    """Text export of the report without Streamlit or HTML."""
    lines = ["SWIMVISION RACE REPORT", report.race_label, "", report.summary, ""]
    if report.lengths:
        lines.append("SPLITS")
        for length in report.lengths:
            rate = f", {length.stroke_rate_spm:.0f} {strokes_word(report.stroke)}/min" if length.stroke_rate_spm else ""
            lines.append(f"- Length {length.number}: {format_time(length.split_s)} (total "
                         f"{format_time(length.cumulative_s)}, pace {format_time(length.pace_per_100_s)} per 100{rate})")
        lines.append("")
    if report.technique:
        lines.append("HOW THE RACE WENT")
        for note in report.technique:
            lines.extend((f"- {note.topic}: {note.observation}", f"  {note.meaning}"))
        lines.append("")
    if report.focus_areas:
        lines.append("WHAT TO WORK ON")
        for area in report.focus_areas:
            lines.extend((f"- {area.title}: {area.why}", f"  {area.how}",
                          f"  You can look up sets for: {', '.join(area.look_up)}"))
        lines.append("")
    lines.extend(f"Note: {note}" for note in report.notes)
    return "\n".join(lines).rstrip() + "\n"


def _render_report(record: AnalysisRecord) -> None:
    report = record.performance
    assert report is not None
    st.success(report.summary)
    cards = []
    if report.race_time_s is not None:
        cards.append(("Official time" if report.race_time_source == "official" else "Race time",
                      format_time(report.race_time_s),
                      "entered by you" if report.race_time_source == "official" else "timed from the video"))
        cards.append((f"Pace per 100 {report.unit}", format_time(report.race_time_s * 100 / report.distance),
                      f"{report.distance / report.race_time_s:.2f} {report.unit}/s"))
    if report.reaction_s is not None:
        cards.append(("Reaction", f"{report.reaction_s:.2f} s", "beep to first movement"))
    if report.average_stroke_rate_spm:
        cards.append(("Stroke rate", f"{report.average_stroke_rate_spm:.0f}",
                      f"{strokes_word(report.stroke)} per minute"))
    if cards:
        for col, (label, value, sub) in zip(st.columns(len(cards)), cards):
            col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)

    tab_race, tab_technique, tab_work, tab_coach = st.tabs(["Race", "Technique", "What to work on", "Ask coach"])
    with tab_race:
        if report.lengths:
            st.dataframe(pd.DataFrame([{
                "Length": length_label(item.number, item.distance, item.unit),
                "Split": format_time(item.split_s),
                "Total": format_time(item.cumulative_s),
                f"Pace / 100 {item.unit}": format_time(item.pace_per_100_s),
                f"Speed ({item.unit}/s)": f"{item.average_speed:.2f}",
                "Stroke rate": f"{item.stroke_rate_spm:.0f}" if item.stroke_rate_spm else "—",
            } for item in report.lengths]), hide_index=True, width="stretch")
            st.bar_chart(pd.DataFrame({
                "Length": [f"Length {item.number}" for item in report.lengths],
                "Split (s)": [item.split_s for item in report.lengths],
            }).set_index("Length"), color="#22D3EE")
            st.caption(f"Stroke rate is {strokes_word(report.stroke)} per minute. The first length includes the dive.")
        else:
            st.info("Not every wall was visible, so per-length splits are not shown.")
        st.dataframe(pd.DataFrame([{
            "Moment": event.label,
            "Race clock": format_time(max(0.0, event.timestamp_s - report.start.timestamp_s)),
            "Time in video": f"{event.timestamp_s:.2f} s",
        } for event in report.events]), hide_index=True, width="stretch")
        for note in report.notes:
            st.caption(note)

    with tab_technique:
        if not report.technique:
            st.info("Not enough of the race was visible to describe how it went.")
        for note in report.technique:
            st.markdown(f"**{note.topic}**")
            st.write(note.observation)
            st.caption(note.meaning)

    with tab_work:
        for number, area in enumerate(report.focus_areas, start=1):
            st.markdown(f"#### {number}. {area.title}")
            st.write(area.why)
            st.write(area.how)
            st.caption("You can look up sets for: " + ", ".join(area.look_up))
        st.download_button("Download report (TXT)", plain_report(report).encode("utf-8"),
                           file_name="swimvision-race-report.txt", mime="text/plain")
        st.download_button("Download analysis data (JSON)", json.dumps(report.to_dict(), indent=2).encode("utf-8"),
                           file_name="swimvision-analysis.json", mime="application/json")

    with tab_coach:
        _render_coach_chat(record.analysis_id, report)


def _render_coach_chat(analysis_id: str, report: RaceAnalysis) -> None:
    st.markdown("#### Ask SwimVision Coach")
    st.write("An AI coach that knows this race. Ask anything about it, or what to do next.")
    key = f"coach-messages-{analysis_id}"
    messages = st.session_state.setdefault(key, [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask your coach a question", key=f"coach-input-{key}")
    if question:
        with st.spinner("Thinking…"):
            try:
                answer = ask_coach(report, question, messages)
            except (CoachUnavailable, SwimVisionError) as exc:
                st.error(str(exc))
                return
        messages.extend(({"role": "user", "content": question}, {"role": "assistant", "content": answer}))
        st.rerun()
