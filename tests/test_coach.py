"""Tests for the user-facing, frontend-neutral coaching response."""

from core.coach import answer_coach_question, build_coach_brief, build_coach_context
from core.race_analysis import (
    RaceAnalysis,
    RaceEvent,
    TechniqueAnalysis,
    TechniqueMetric,
    TrainingSet,
)


def _report() -> RaceAnalysis:
    event = RaceEvent("start", "Estimated takeoff", 10, 1.0, "medium", "movement")
    finish = RaceEvent("finish", "Estimated finish", 600, 60.0, "medium", "wall stop")
    good = TechniqueMetric(
        "stroke_rhythm", "Stroke rhythm consistency", "good", "medium", "8% variation",
        "Your stroke timing stayed repeatable.", "Keep the same rhythm when speed increases.",
        "Six visible cycles",
    )
    improve = TechniqueMetric(
        "vertical_path", "Vertical body-path control", "improve", "medium", "14% body height",
        "Your body moved up and down more as the swim continued.",
        "Try side-kick and six-kick-switch while keeping your head still.", "Tracked centre path",
    )
    unavailable = TechniqueMetric(
        "kick", "Kick timing", "not_measurable", "low", "Not measured",
        "The legs were underwater.", "Use an underwater side view.", "No ankle keypoints",
    )
    technique = TechniqueAnalysis(
        "Start with vertical body-path control.", (good, improve, unavailable),
        (good.label, improve.label), (unavailable.label,),
    )
    practice = TrainingSet(
        "Body-line control", "Stay level", "6 × 25 easy-perfect, 20 seconds rest.",
        "Keep the head still on every repeat.",
    )
    return RaceAnalysis(
        race_label="100 yd Freestyle (SCY)", status="review_needed", start=event, finish=finish,
        events=(event, finish), lengths=(), race_time_s=58.21, race_time_source="official_manual",
        estimated_race_time_s=59.0, stroke_cycles=20, arm_strokes=40,
        average_stroke_rate_spm=42.0, track_observed_fraction=0.98, findings=(),
        technique=technique, training_sets=(practice,), nutrition_tips=("Drink water after practice.",),
        limitations=("Kick timing was not measured.",),
    )


def test_brief_is_plain_and_frontend_ready():
    brief = build_coach_brief(_report())
    payload = brief.to_dict()
    assert brief.time_label == "Official race time"
    assert brief.race_time_display == "58.21 s"
    assert brief.strengths[0].title == "Stroke rhythm consistency"
    assert brief.priorities[0].title == "Vertical body-path control"
    assert "stable technique signal" not in str(payload).lower()
    assert "Kick timing" not in [point["title"] for point in payload["priorities"]]


def test_coach_answers_only_from_report():
    report = _report()
    assert "58.21 s" in answer_coach_question(report, "What was my time?")
    assert "vertical body-path control" in answer_coach_question(report, "What should I fix first?").lower()
    assert "6 × 25" in answer_coach_question(report, "Give me a practice set")
    confidence = answer_coach_question(report, "How accurate is this?")
    assert "medium confidence" in confidence


def test_local_model_context_marks_unmeasured_details_as_limits():
    context = build_coach_context(_report())
    assert "Never invent" in context
    assert '"Kick timing"' not in context.split('"measured_technique"', 1)[1].split('"training_sets"', 1)[0]
    assert "Kick timing was not measured" in context
