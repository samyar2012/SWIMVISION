"""User-facing coaching presentation built from a measured race report.

The race-analysis module owns measurements.  This module only translates those
measurements into plain language, which keeps the same output usable by the
current Streamlit prototype and a future React client.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Callable

from core.race_analysis import RaceAnalysis, TechniqueMetric, TrainingSet

COACH_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CoachPoint:
    title: str
    explanation: str
    action: str
    confidence: str


@dataclass(frozen=True)
class CoachBrief:
    """Small, JSON-ready response intended for a non-technical interface."""

    race_label: str
    time_label: str
    race_time_s: float
    race_time_display: str
    video_estimate_s: float
    video_estimate_display: str
    headline: str
    overview: str
    strengths: tuple[CoachPoint, ...]
    priorities: tuple[CoachPoint, ...]
    next_practice: TrainingSet | None
    suggested_questions: tuple[str, ...]
    format_version: int = COACH_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_seconds(value: float) -> str:
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes}:{seconds:05.2f}" if minutes else f"{seconds:.2f} s"


def _point(metric: TechniqueMetric) -> CoachPoint:
    return CoachPoint(
        title=metric.label,
        explanation=metric.explanation,
        action=metric.action,
        confidence=metric.confidence,
    )


def build_coach_brief(report: RaceAnalysis) -> CoachBrief:
    """Create the small response a family-facing UI should show first."""
    measured = [m for m in report.technique.metrics if m.rating != "not_measurable"]
    strengths = tuple(_point(m) for m in measured if m.rating == "good")[:2]
    improvement = [m for m in measured if m.rating == "improve"]
    review = [m for m in measured if m.rating == "check"]
    priorities = tuple(_point(m) for m in [*improvement, *review][:2])

    if priorities:
        headline = f"Start with {priorities[0].title.lower()}."
        overview = (
            f"Your clearest next step is {priorities[0].title.lower()}. "
            f"{priorities[0].action}"
        )
    elif strengths:
        headline = f"{strengths[0].title} was your clearest strength."
        overview = (
            "The visible parts of this swim were controlled. "
            f"Keep building on {strengths[0].title.lower()}."
        )
    else:
        headline = "This video is a starting point, not a technique grade."
        overview = (
            "The camera did not show enough repeatable movement for a confident technique call. "
            "Use the filming advice below and try again."
        )

    time_label = "Official race time" if report.race_time_source == "official_manual" else "Estimated race time"
    return CoachBrief(
        race_label=report.race_label,
        time_label=time_label,
        race_time_s=report.race_time_s,
        race_time_display=format_seconds(report.race_time_s),
        video_estimate_s=report.estimated_race_time_s,
        video_estimate_display=format_seconds(report.estimated_race_time_s),
        headline=headline,
        overview=overview,
        strengths=strengths,
        priorities=priorities,
        next_practice=report.training_sets[0] if report.training_sets else None,
        suggested_questions=(
            "What should I fix first?",
            "Give me a practice set.",
            "What did I do well?",
            "How confident is this analysis?",
        ),
    )


def build_coach_context(report: RaceAnalysis) -> str:
    """Return grounded JSON context for an optional local language model.

    A future backend may pass this text to a small local model.  The instruction
    explicitly prevents the model from inventing measurements that are not in
    the report.
    """
    payload = {
        "instructions": (
            "Answer as a supportive swim coach in plain language. Use only the supplied report. "
            "Never invent joint angles, breathing, kick, splits, or measurements. Say when the "
            "video cannot answer. Keep the answer under 140 words."
        ),
        "coach_brief": build_coach_brief(report).to_dict(),
        "measured_technique": [
            asdict(metric)
            for metric in report.technique.metrics
            if metric.rating != "not_measurable"
        ],
        "training_sets": [asdict(item) for item in report.training_sets],
        "nutrition_tips": list(report.nutrition_tips),
        "limitations": list(report.limitations),
    }
    return json.dumps(payload, ensure_ascii=False)


def answer_coach_question(
    report: RaceAnalysis,
    question: str,
    model: Callable[[str], str] | None = None,
) -> str:
    """Answer common questions locally, or use an injected lightweight model.

    The deterministic fallback keeps the feature deployable without model
    downloads.  Passing a callable later lets a backend use Ollama, llama.cpp,
    or another hosted/local provider without changing the React response shape.
    """
    question = question.strip()
    if not question:
        return "Ask me what to fix, what went well, or what to practice next."
    if model is not None:
        prompt = f"{build_coach_context(report)}\n\nSwimmer question: {question}"
        return model(prompt).strip()

    brief = build_coach_brief(report)
    lowered = question.casefold()
    if any(word in lowered for word in ("time", "fast", "result")):
        answer = f"Your {brief.time_label.lower()} is {brief.race_time_display}."
        if report.race_time_source == "official_manual":
            answer += f" The separate video estimate is {brief.video_estimate_display}."
        return answer

    if any(word in lowered for word in ("fix", "improve", "wrong", "technique", "first")):
        if not brief.priorities:
            return brief.overview
        point = brief.priorities[0]
        return f"Work on {point.title.lower()} first. {point.explanation} Try this: {point.action}"

    if any(word in lowered for word in ("practice", "set", "drill", "train")):
        item = brief.next_practice
        if item is None:
            return "This report did not produce a trustworthy practice recommendation yet."
        return f"Try {item.name}: {item.set_text} Success looks like: {item.success_check}"

    if any(word in lowered for word in ("good", "strength", "well", "best")):
        if not brief.strengths:
            return (
                "This video did not show a clear measurable strength yet. That does not mean your "
                "technique was bad; it means the camera evidence was limited."
            )
        point = brief.strengths[0]
        return f"Your clearest strength was {point.title.lower()}. {point.explanation}"

    if any(word in lowered for word in ("confident", "confidence", "evidence", "know", "accurate")):
        if brief.priorities:
            point = brief.priorities[0]
            return (
                f"The {point.title.lower()} recommendation has {point.confidence} confidence. "
                "It comes from movement visible in this video, not a full joint-by-joint pose model."
            )
        return (
            "The report only states what the video supports. Details such as catch angle, breathing "
            "angle, and kick mechanics are withheld when they cannot be measured reliably."
        )

    if any(word in lowered for word in ("food", "eat", "diet", "fuel", "recover")):
        tips = " ".join(report.nutrition_tips[:2])
        return f"{tips} This is general sports guidance, not a personal medical or diet prescription."

    return f"{brief.overview} You can ask what to fix first, what went well, or for a practice set."
