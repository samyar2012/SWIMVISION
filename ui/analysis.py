"""Step 4: run and explain the race analysis in plain language."""

from __future__ import annotations

import json
import logging

import pandas as pd
import streamlit as st

from core.coach import answer_coach_question, build_coach_brief
from core.errors import SwimVisionError
from core.race_analysis import Finding, RaceAnalysis, analyze_race
from storage.analyses import AnalysisRecord, load_tracking_table, save_analysis
from ui.styles import metric_card, step_label

logger = logging.getLogger(__name__)


def render_analysis(record: AnalysisRecord) -> None:
    step_label("Step 4 · Understand your race")
    st.write("See what went well, what to improve first, and what to practice next.")
    official_time_s = _official_time_input(record)
    if record.performance is None:
        if st.button("Analyze my race", type="primary", width="stretch"):
            if official_time_s is not False:
                _run_analysis(record, official_time_s)
        return
    _render_report(record.performance)
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
    report = record.performance
    current = report.race_time_s if report and report.race_time_source == "official_manual" else None
    with st.expander("Add my official race time (optional)", expanded=current is not None):
        st.caption("This only improves the report. It does not stop the video, change tracking, or choose a frame.")
        use_official = st.checkbox(
            "Use a time from the scoreboard or stopwatch",
            value=current is not None,
            key=f"official-time-enabled-{record.analysis_id}",
        )
        value = st.text_input(
            "Official race time",
            value=_format_seconds_input(current) if current is not None else "",
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
    bar = st.progress(0.0, text="Reading the swimmer's movement…")

    def progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Analyzing movement frame {done} of {total}")

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
    save_analysis(record)
    st.rerun()


def _format_seconds(value: float) -> str:
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes}:{seconds:05.2f}" if minutes else f"{seconds:.2f} s"


def _format_seconds_input(value: float) -> str:
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes}:{seconds:05.2f}" if minutes else f"{seconds:.2f}"


def _plain_report(report: RaceAnalysis) -> str:
    """Family/coach-friendly text export without Streamlit or HTML."""
    brief = build_coach_brief(report)
    lines = [
        "SWIMVISION RACE REPORT",
        report.race_label,
        "",
        f"Race time: {_format_seconds(report.race_time_s)} ({report.race_time_source.replace('_', ' ')})",
        f"Automatic video estimate: {_format_seconds(report.estimated_race_time_s)}",
        f"Direct video evidence: {report.track_observed_fraction:.0%}",
        f"Measured lengths: {len(report.lengths)}",
        f"Estimated stroke cycles: {report.stroke_cycles if report.stroke_cycles is not None else 'not reported'}",
        "",
        "KEY FINDINGS",
    ]
    for finding in report.findings:
        lines.extend((f"- {finding.title}: {finding.explanation}", f"  Next step: {finding.action}"))
    lines.extend(("", "TECHNIQUE", brief.headline, brief.overview))
    if brief.strengths:
        lines.append("What looked good:")
        for point in brief.strengths:
            lines.append(f"- {point.title}: {point.explanation}")
    if brief.priorities:
        lines.append("What to work on first:")
        for point in brief.priorities:
            lines.extend((f"- {point.title}: {point.explanation}", f"  Try this: {point.action}"))
    lines.extend(("", "EXAMPLE TRAINING SETS"))
    for item in report.training_sets:
        lines.extend((f"- {item.name}: {item.set_text}", f"  Success check: {item.success_check}"))
    lines.extend(("", "FUEL AND RECOVERY"))
    lines.extend(f"- {tip}" for tip in report.nutrition_tips)
    lines.extend(("", "IMPORTANT LIMITATIONS"))
    lines.extend(f"- {item}" for item in report.limitations)
    lines.extend(("", "This report supports coaching review; it is not medical or dietary advice."))
    return "\n".join(lines) + "\n"


def _render_finding(finding: Finding) -> None:
    icon = {"good": "✅", "improve": "🎯", "check": "🔎"}.get(finding.tone, "•")
    st.markdown(f"**{icon} {finding.title}**")
    st.write(finding.explanation)
    st.caption(f"What to do: {finding.action}")


def _render_report(report: RaceAnalysis) -> None:
    brief = build_coach_brief(report)
    st.success(brief.headline)
    if report.status == "review_needed":
        st.caption("Some video measurements are estimates. Open ‘How this was measured’ only if you want the details.")
    with st.expander("How this was measured", expanded=False):
        st.write("**Verified:** video timing, frame numbers, selected-swimmer path, tracking status and expected race structure.")
        st.write("**Estimated:** takeoff, water entry, turns, breakouts, stopped-at-wall finish and any splits built from those events.")
        st.write("**Experimental:** stroke cycles/rate and technique proxies from temporal motion. Use the evidence before coaching from them.")
        st.write("**Not guessed:** catch/elbow angles, breathing angle and kick mechanics when anatomical keypoints are unavailable.")
    time_label = "Official time" if report.race_time_source == "official_manual" else "Estimated time"
    cards = [
        (time_label, _format_seconds(report.race_time_s),
         "entered by user" if report.race_time_source == "official_manual"
         else f"start: {report.start.confidence} · finish: {report.finish.confidence}"),
        ("Lengths measured", f"{len(report.lengths)}", "shown only when all expected turns are found"),
        ("Stroke cycles", str(report.stroke_cycles) if report.stroke_cycles is not None else "—",
         "motion estimate; review before using"),
        ("Direct video evidence", f"{report.track_observed_fraction:.0%}", "remaining frames use explicit estimates"),
    ]
    for col, (label, value, sub) in zip(st.columns(4), cards):
        col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)

    tab_summary, tab_technique, tab_splits, tab_plan, tab_food, tab_coach, tab_evidence = st.tabs(
        ["Summary", "Technique", "Splits & strokes", "Training plan", "Fuel & recovery", "Ask coach", "Evidence"])
    with tab_summary:
        st.markdown(f"#### {brief.headline}")
        st.write(brief.overview)
        if brief.next_practice:
            st.markdown("#### Your next practice")
            st.write(f"**{brief.next_practice.name}:** {brief.next_practice.set_text}")
            st.caption(f"Success looks like: {brief.next_practice.success_check}")
        with st.expander("More race takeaways", expanded=False):
            for finding in report.findings:
                _render_finding(finding)

    with tab_technique:
        st.markdown("#### Your technique, in plain language")
        if brief.strengths:
            st.markdown("##### What looked good")
            for point in brief.strengths:
                st.write(f"**✅ {point.title}**")
                st.write(point.explanation)
        else:
            st.info("This angle did not show a clear, repeatable strength. That does not mean your technique was bad.")

        st.markdown("##### What to work on first")
        if brief.priorities:
            for number, point in enumerate(brief.priorities, start=1):
                st.write(f"**{number}. {point.title}**")
                st.write(point.explanation)
                st.write(f"Try this: {point.action}")
        else:
            st.write("No clear technique problem was measurable in this video.")

        with st.expander("See measurements and camera limits", expanded=False):
            for metric in report.technique.metrics:
                if metric.rating == "not_measurable":
                    continue
                st.write(f"**{metric.label}:** {metric.value}")
                st.caption(f"Confidence: {metric.confidence} · {metric.evidence}")
            if report.technique.not_measurable:
                st.write("**This camera could not reliably judge:** " + ", ".join(report.technique.not_measurable))
                st.caption("SwimVision leaves these blank instead of guessing.")

    with tab_splits:
        if report.lengths:
            rows = [{
                "Length": item.number,
                "Stroke": item.stroke,
                "Distance": f"{item.distance} {item.unit}",
                "Split": _format_seconds(item.split_s),
                f"Speed ({item.unit}/s)": item.average_speed,
                "Cycles": item.stroke_cycles if item.stroke_cycles is not None else "—",
                "Cycles/min": item.stroke_rate_spm if item.stroke_rate_spm is not None else "—",
                f"Length-normalized {item.unit}/cycle": item.distance_per_cycle if item.distance_per_cycle is not None else "—",
                "Confidence": item.confidence,
            } for item in report.lengths]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            chart_data = pd.DataFrame({
                "Length": [f"L{item.number}" for item in report.lengths],
                "Split (s)": [item.split_s for item in report.lengths],
            }).set_index("Length")
            st.markdown("##### Pace by length")
            st.bar_chart(chart_data, color="#22D3EE")
            rate_rows = [item for item in report.lengths if item.stroke_rate_spm is not None]
            if rate_rows:
                st.markdown("##### Estimated cycle rate by length")
                st.line_chart(pd.DataFrame({
                    "Length": [f"L{item.number}" for item in rate_rows],
                    "Cycles/min": [item.stroke_rate_spm for item in rate_rows],
                }).set_index("Length"), color="#F59E0B")
            if report.arm_strokes is not None and report.stroke_cycles is not None and report.arm_strokes != report.stroke_cycles:
                st.caption(f"Freestyle/backstroke convention: {report.stroke_cycles} same-arm cycles ≈ "
                           f"{report.arm_strokes} individual arm strokes.")
        else:
            st.info("Per-length splits are hidden because the system did not find every expected turn. It will not invent them.")

    with tab_plan:
        st.markdown("#### Example sets to discuss with your coach")
        for number, item in enumerate(report.training_sets, start=1):
            with st.expander(f"{number}. {item.name} — {item.purpose}", expanded=number == 1):
                st.write(item.set_text)
                st.caption(f"Success looks like: {item.success_check}")
        st.info("These are technique-focused examples, not a complete training program. Adjust volume and rest with a qualified coach.")

    with tab_food:
        st.markdown("#### Simple fueling and recovery reminders")
        for tip in report.nutrition_tips:
            st.write(f"• {tip}")
        st.warning("SwimVision does not prescribe calories, weight change, supplements or a medical diet. Young athletes should make nutrition decisions with a parent/guardian and a qualified sports dietitian.")
        st.caption("Guidance basis: USA Swimming nutrition resources, USOPC's food-first Athlete Plate, and USADA's athlete nutrition guide.")

    with tab_coach:
        _render_coach_chat(report)

    with tab_evidence:
        if report.race_time_source == "official_manual":
            st.write(f"**Official time entered by user:** {_format_seconds(report.race_time_s)}")
            st.caption(f"Independent automatic video estimate: {_format_seconds(report.estimated_race_time_s)}")
        st.caption("Events are ordered by their exact source frame. Confidence describes the event detector—not the swimmer's performance.")
        st.markdown("#### Detected race events")
        st.dataframe(pd.DataFrame([{
            "Event": e.label, "Time in video": _format_seconds(e.timestamp_s), "Frame": e.frame_index,
            "Confidence": e.confidence, "Why it was marked": e.evidence,
        } for e in report.events]), hide_index=True, width="stretch")
        st.markdown("#### What is not fully measured yet")
        for limitation in report.limitations:
            st.write(f"• {limitation}")
        st.download_button(
            "Download app-ready coach summary (JSON)",
            json.dumps(brief.to_dict(), indent=2).encode("utf-8"),
            file_name="swimvision-coach-summary.json", mime="application/json",
        )
        st.download_button(
            "Download complete technical analysis (JSON)",
            json.dumps(report.to_dict(), indent=2).encode("utf-8"),
            file_name="swimvision-analysis.json", mime="application/json",
        )
        st.download_button(
            "Download family/coach report (TXT)",
            _plain_report(report).encode("utf-8"),
            file_name="swimvision-coach-report.txt", mime="text/plain",
        )
        if report.lengths:
            st.download_button(
                "Download length metrics (CSV)",
                pd.DataFrame([item.__dict__ for item in report.lengths]).to_csv(index=False).encode("utf-8"),
                file_name="swimvision-lengths.csv", mime="text/csv",
            )


def _render_coach_chat(report: RaceAnalysis) -> None:
    st.markdown("#### Ask SwimVision Coach")
    st.write("Ask what to fix, what went well, or what to practice next. Answers stay inside this race report.")
    key = f"coach-messages-{report.race_label}-{report.start.frame_index}-{report.finish.frame_index}"
    messages = st.session_state.setdefault(key, [])
    if not messages:
        messages.append({"role": "assistant", "content": build_coach_brief(report).overview})
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ask your coach a question", key=f"coach-input-{key}")
    if question:
        messages.append({"role": "user", "content": question})
        answer = answer_coach_question(report, question)
        messages.append({"role": "assistant", "content": answer})
        st.rerun()
