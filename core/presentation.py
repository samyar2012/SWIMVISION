"""JSON presentation layer for the React application.

This module does not create measurements. It turns the race analysis into a
stable, readable response: an overview, a technique breakdown in plain
language, the real splits with pace, and what to work on next.
"""

from __future__ import annotations

from typing import Any

from core.race_analysis import RaceAnalysis
from core.race_report import format_time, length_label, strokes_word

_UNIT_WORD = {"yd": "yards", "m": "meters"}

_TIME_SOURCE_NOTES = {
    "official": "Official time you entered",
    "video": "Timed from the video",
    "unknown": "The finish was not visible",
}


def report_payload(report: RaceAnalysis) -> dict[str, Any]:
    """Return the response consumed by the React client."""
    unit_word = _UNIT_WORD.get(report.unit, report.unit)
    stroke_unit = f"{strokes_word(report.stroke)} per minute"
    race_start = report.start.timestamp_s

    highlights: list[dict[str, str]] = []
    if report.race_time_s is not None:
        highlights.append({"label": "Race time", "value": format_time(report.race_time_s),
                           "detail": _TIME_SOURCE_NOTES[report.race_time_source]})
        highlights.append({"label": f"Pace per 100 {report.unit}",
                           "value": format_time(report.race_time_s * 100.0 / report.distance),
                           "detail": f"{report.distance / report.race_time_s:.2f} {unit_word} per second"})
    if report.reaction_s is not None:
        highlights.append({"label": "Reaction", "value": f"{report.reaction_s:.2f} s",
                           "detail": "Beep to first movement"})
    if report.average_stroke_rate_spm:
        highlights.append({"label": "Stroke rate", "value": f"{report.average_stroke_rate_spm:.0f}",
                           "detail": stroke_unit})

    splits = []
    previous_split = None
    for length in report.lengths:
        delta = None if previous_split is None else round(length.split_s - previous_split, 2)
        splits.append({
            "number": length.number,
            "label": length_label(length.number, length.distance, length.unit),
            "distance_label": f"{length.distance * length.number} {length.unit}",
            "split": format_time(length.split_s),
            "cumulative": format_time(length.cumulative_s),
            "pace_per_100": format_time(length.pace_per_100_s),
            "speed": f"{length.average_speed:.2f} {length.unit}/s",
            "stroke_rate": f"{length.stroke_rate_spm:.0f}" if length.stroke_rate_spm else None,
            "split_s": length.split_s,
            "delta_s": delta,
            "video_start_s": length.start_s,
            "video_end_s": length.end_s,
        })
        previous_split = length.split_s
    fastest = min(report.lengths, key=lambda item: item.split_s).number if len(report.lengths) > 1 else None

    moments = [
        {
            "label": event.label,
            "video_s": event.timestamp_s,
            "race_clock": format_time(max(0.0, event.timestamp_s - race_start)) if event.kind != "start" else "0.00",
        }
        for event in report.events
    ]

    return {
        "race_label": report.race_label,
        "stroke": report.stroke,
        "distance": report.distance,
        "unit": report.unit,
        "pool_length": report.pool_length,
        "complete": report.status == "complete",
        "race_time": format_time(report.race_time_s) if report.race_time_s is not None else None,
        "race_time_source": report.race_time_source,
        "overview": report.summary,
        "highlights": highlights,
        "technique": [
            {"topic": note.topic, "tone": note.tone, "observation": note.observation, "meaning": note.meaning}
            for note in report.technique
        ],
        "splits": splits,
        "fastest_length": fastest,
        "expected_lengths": report.expected_lengths,
        "stroke_rate_unit": stroke_unit,
        "moments": moments,
        "focus": [
            {"title": area.title, "why": area.why, "how": area.how, "look_up": list(area.look_up)}
            for area in report.focus_areas
        ],
        "notes": list(report.notes),
        "suggested_questions": _suggested_questions(report),
    }


def _suggested_questions(report: RaceAnalysis) -> list[str]:
    questions = []
    if len(report.lengths) >= 2:
        questions.append("Why was my second length slower?")
    if report.reaction_s is not None:
        questions.append("How can I react faster to the beep?")
    if report.focus_areas:
        questions.append(f"How do I work on {report.focus_areas[0].title.lower()}?")
    questions.append(f"What should I focus on in my next {report.stroke.lower()} race?")
    return questions[:4]
