"""JSON presentation layer for the family-facing React application.

This module does not create new measurements. It turns the evidence-backed
race report into a stable, readable API response and keeps developer details
out of the interface.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from core.coach import build_coach_brief, format_seconds
from core.race_analysis import RaceAnalysis

_TECHNIQUE_TITLES = {
    "Freestyle": "Balance, rhythm, and finish",
    "Backstroke": "Body line, rhythm, and wall work",
    "Breaststroke": "Pull-kick rhythm, glide, and walls",
    "Butterfly": "Rhythm, body path, and finish",
    "Individual Medley": "Transitions, control, and finish",
}

_STROKE_COPY: dict[str, dict[str, tuple[str, str]]] = {
    "Freestyle": {
        "stroke_rhythm": ("Alternating arm rhythm", "Swim 6 × 25 smooth-fast with a tempo cue; keep the left-right rhythm even as speed rises."),
        "vertical_path": ("Body-line steadiness", "Use side-kick into six-kick-switch and keep the crown of the head moving straight down the lane."),
        "cycle_economy": ("Freestyle length under fatigue", "Count cycles on 6 × 25 at race pace and keep the final repeat within one cycle of the first."),
        "turn_consistency": ("Flip-turn and breakout repeatability", "Rehearse the same last stroke, tight flip, push depth, and first stroke on every wall."),
        "finish_execution": ("Freestyle finish timing", "Accelerate the final strokes and touch at full reach instead of gliding into the wall."),
    },
    "Backstroke": {
        "stroke_rhythm": ("Backstroke arm rhythm", "Use 6 × 25 with a steady tempo and make each hand entry match the timing of the other side."),
        "vertical_path": ("Surface-line steadiness", "Use side-kick and single-arm backstroke while keeping the head quiet and the body high."),
        "cycle_economy": ("Backstroke length under fatigue", "Count cycles on race-pace 25s and hold distance per cycle without slowing the tempo."),
        "turn_consistency": ("Backstroke turn and breakout", "Repeat the same stroke count into the flags, rotation, push depth, and breakout on every wall."),
        "finish_execution": ("Backstroke finish timing", "Know the stroke count from the flags and reach through the touch without gliding or turning over."),
    },
    "Breaststroke": {
        "stroke_rhythm": ("Pull-kick-glide rhythm", "Swim 6 × 25 as pull, breathe, kick, line—keeping the same timing instead of rushing the recovery."),
        "vertical_path": ("Breaststroke rise-and-fall pattern", "Use 2-kicks-1-pull drill and send the body forward after each kick instead of only moving upward."),
        "cycle_economy": ("Distance per breaststroke cycle", "Count cycles on 6 × 25 and protect the streamlined position as fatigue builds."),
        "turn_consistency": ("Two-hand turn and pullout", "Practice a quick two-hand touch, compact feet, strong streamline, then the same pullout timing every wall."),
        "finish_execution": ("Breaststroke finish timing", "Time the last full cycle so both hands reach the wall together without a long glide."),
    },
    "Butterfly": {
        "stroke_rhythm": ("Butterfly cycle rhythm", "Use 6 × 25 of 3-3-3 drill into full stroke and keep the kick-arm timing repeatable."),
        "vertical_path": ("Butterfly body-wave pattern", "Use body-dolphin and single-arm fly to send the chest forward while keeping the wave low and continuous."),
        "cycle_economy": ("Butterfly length under fatigue", "Count cycles on short quality repeats and stop before the recovery shortens or stalls."),
        "turn_consistency": ("Butterfly turn and breakout", "Rehearse the two-hand touch, fast feet, push, dolphin kicks, and first stroke as one sequence."),
        "finish_execution": ("Butterfly finish timing", "Choose the final full stroke early enough to reach a simultaneous two-hand touch without gliding."),
    },
    "Individual Medley": {
        "stroke_rhythm": ("Rhythm through all four strokes", "Use short IM-order repeats and keep each stroke controlled before increasing transition speed."),
        "vertical_path": ("Body-line control across strokes", "Film and review each stroke leg separately; use the stroke-specific body-line cue for that leg."),
        "cycle_economy": ("Cycle length through the IM", "Count each stroke by length and protect the final freestyle length when fatigue is highest."),
        "turn_consistency": ("IM transition walls", "Practice the legal finish for one stroke and the first movement of the next as one connected skill."),
        "finish_execution": ("IM freestyle finish", "Build the final five metres and touch at full reach instead of waiting for the wall."),
    },
}

_STROKE_PRACTICE = {
    "Freestyle": ("Freestyle line + tempo", "8 × 25: 12.5 side-kick or six-kick-switch, then 12.5 smooth-fast freestyle; 20 seconds rest.", "Head stays quiet and stroke count stays within one cycle."),
    "Backstroke": ("Backstroke line + rhythm", "8 × 25: 6 kicks on one side, switch, then finish smooth-fast backstroke; 20 seconds rest.", "Entries stay even and the body remains high through the finish."),
    "Breaststroke": ("Breaststroke timing", "8 × 25 alternating 2-kicks-1-pull and full stroke; 25 seconds rest.", "Each kick finishes in a narrow line before the next pull."),
    "Butterfly": ("Butterfly rhythm", "8 × 25 as 3 right-arm, 3 left-arm, 3 full strokes, then easy to the wall; 30 seconds rest.", "The second kick connects to the hand exit without a pause."),
    "Individual Medley": ("IM transitions", "8 × 25 in IM order, focusing on the legal finish and first movement of the next stroke; 25 seconds rest.", "Every transition keeps momentum and stays legal."),
}


def _stroke_name(report: RaceAnalysis) -> str:
    # Race labels are written as "50 yd Freestyle (SCY)".
    for stroke in _TECHNIQUE_TITLES:
        if stroke in report.race_label:
            return stroke
    return "Swimming"


def report_payload(report: RaceAnalysis) -> dict[str, Any]:
    """Return the non-technical response consumed by the React client."""
    brief = build_coach_brief(report).to_dict()
    stroke = _stroke_name(report)
    visible = [metric for metric in report.technique.metrics if metric.rating != "not_measurable"]
    copy = _STROKE_COPY.get(stroke, {})

    def point(metric: Any) -> dict[str, Any]:
        label, action = copy.get(metric.key, (metric.label, metric.action))
        return {"title": label, "explanation": metric.explanation,
                "action": action, "confidence": metric.confidence}

    strengths = [point(metric) for metric in visible if metric.rating == "good"][:2]
    priorities = [point(metric) for metric in visible if metric.rating in {"improve", "check"}][:2]
    if priorities:
        brief["headline"] = f"Start with {priorities[0]['title'].lower()}."
        brief["overview"] = f"{priorities[0]['explanation']} {priorities[0]['action']}"
    elif strengths:
        brief["headline"] = f"{strengths[0]['title']} was your clearest strength."
        brief["overview"] = f"{strengths[0]['explanation']} {strengths[0]['action']}"
    brief["strengths"] = strengths
    brief["priorities"] = priorities

    training_sets = [asdict(item) for item in report.training_sets]
    if practice := _STROKE_PRACTICE.get(stroke):
        name, set_text, success = practice
        training_sets.insert(0, {
            "name": name,
            "purpose": f"Build the {stroke.lower()} skill visible in this race",
            "set_text": set_text,
            "success_check": success,
        })
        training_sets = training_sets[:3]
        brief["next_practice"] = training_sets[0]

    brief.update({
        "lengths": [
            {
                **asdict(length),
                "split_display": format_seconds(length.split_s),
                "pace_note": (
                    f"{length.average_speed:.2f} {length.unit}/s"
                    if length.average_speed > 0 else None
                ),
            }
            for length in report.lengths
        ],
        "training_sets": training_sets,
        "nutrition_tips": list(report.nutrition_tips[:4]),
        "technique": {
            "stroke": stroke,
            "title": _TECHNIQUE_TITLES.get(stroke, "Technique breakdown"),
            "summary": report.technique.summary,
            "visible_cycles": report.stroke_cycles,
            "sections": [
                {
                    "title": copy.get(metric.key, (metric.label, metric.action))[0],
                    "rating": metric.rating,
                    "rating_label": {
                        "good": "Working well",
                        "improve": "Priority",
                        "check": "Review",
                    }.get(metric.rating, "Review"),
                    "explanation": metric.explanation,
                    "action": copy.get(metric.key, (metric.label, metric.action))[1],
                    "confidence": metric.confidence,
                }
                for metric in visible
            ],
            "camera_note": (
                "SwimVision only scores movement this camera shows repeatedly. "
                "It does not guess hidden joint angles, breathing, or underwater kick details."
            ),
        },
    })
    return brief
