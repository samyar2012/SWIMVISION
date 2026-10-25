"""Step 4: run and explain the race analysis in plain language."""

from __future__ import annotations

import json
import logging

import pandas as pd
import streamlit as st

from core.errors import SwimVisionError
from core.race_analysis import Finding, RaceAnalysis, analyze_race
from storage.analyses import AnalysisRecord, load_tracking_table, save_analysis
from ui.styles import metric_card, step_label

logger = logging.getLogger(__name__)


def render_analysis(record: AnalysisRecord) -> None:
    step_label("Step 4 · Understand your race")
    st.write("Turn the verified swimmer path into race events, length splits, stroke rhythm and an improvement plan.")
    if record.performance is None:
        if st.button("Analyze my race", type="primary", width="stretch"):
            _run_analysis(record)
        return
    _render_report(record.performance)
    if st.button("Re-analyze this race", width="stretch"):
        _run_analysis(record)


def _run_analysis(record: AnalysisRecord) -> None:
    table = load_tracking_table(record)
    if table is None:
        st.error("The tracking measurements are missing. Run tracking again first.")
        return
    bar = st.progress(0.0, text="Reading the swimmer's movement…")

    def progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Analyzing movement frame {done} of {total}")

    try:
        record.performance = analyze_race(record.absolute_video_path(), table, record.race, record.video, progress)
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


def _plain_report(report: RaceAnalysis) -> str:
    """Family/coach-friendly text export without Streamlit or HTML."""
    lines = [
        "SWIMVISION RACE REPORT",
        report.race_label,
        "",
        f"Provisional race time: {_format_seconds(report.race_time_s)}",
        f"Direct video evidence: {report.track_observed_fraction:.0%}",
        f"Measured lengths: {len(report.lengths)}",
        f"Estimated stroke cycles: {report.stroke_cycles if report.stroke_cycles is not None else 'not reported'}",
        "",
        "KEY FINDINGS",
    ]
    for finding in report.findings:
        lines.extend((f"- {finding.title}: {finding.explanation}", f"  Next step: {finding.action}"))
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
    st.success("Ready to use: selected-swimmer tracking, full-race event timeline, expected-turn checks and evidence export.")
    if report.status == "review_needed":
        st.warning("Review before presenting as exact: takeoff, finish, breakout and stroke-cycle values are automatic estimates. Low-confidence numbers are marked.")
    with st.expander("What is verified, estimated, and experimental", expanded=False):
        st.write("**Verified:** video timing, frame numbers, selected-swimmer path, tracking status and expected race structure.")
        st.write("**Estimated:** takeoff, water entry, turns, breakouts, finish and any splits built from those events.")
        st.write("**Experimental:** stroke cycles/rate from crop motion. Use the evidence tab to review before coaching from it.")
    cards = [
        ("Provisional time", _format_seconds(report.race_time_s),
         f"start: {report.start.confidence} · finish: {report.finish.confidence}"),
        ("Lengths measured", f"{len(report.lengths)}", "shown only when all expected turns are found"),
        ("Stroke cycles", str(report.stroke_cycles) if report.stroke_cycles is not None else "—",
         "motion estimate; review before using"),
        ("Direct video evidence", f"{report.track_observed_fraction:.0%}", "remaining frames use explicit estimates"),
    ]
    for col, (label, value, sub) in zip(st.columns(4), cards):
        col.markdown(metric_card(label, value, sub), unsafe_allow_html=True)

    tab_summary, tab_splits, tab_plan, tab_food, tab_evidence = st.tabs(
        ["Summary", "Splits & strokes", "Training plan", "Fuel & recovery", "Evidence"])
    with tab_summary:
        st.markdown("#### Your clearest takeaways")
        for finding in report.findings:
            _render_finding(finding)

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

    with tab_evidence:
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
            "Download complete analysis (JSON)",
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
