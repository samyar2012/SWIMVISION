"""Turn race measurements into plain-language feedback a swimmer can act on.

Every sentence is derived from a number the video actually produced. When a
measurement is missing, the matching feedback is left out rather than guessed.
"""

from __future__ import annotations

from dataclasses import replace

from core.race_analysis import FocusArea, LengthMetrics, RaceAnalysis, TechniqueNote

_UNIT_WORD = {"yd": "yards", "m": "meters"}
# The first length includes the dive, so it is normally this much faster.
_DIVE_ALLOWANCE = 0.08


def format_time(seconds: float | None) -> str:
    """``31.0`` -> ``"31.00"``, ``62.45`` -> ``"1:02.45"``."""
    if seconds is None:
        return "—"
    seconds = round(seconds, 2)
    if seconds < 60:
        return f"{seconds:.2f}"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:05.2f}"


def strokes_word(stroke: str) -> str:
    return "arm strokes" if stroke in ("Freestyle", "Backstroke") else "strokes"


def length_label(number: int, pool_length: int, unit: str) -> str:
    """``1, 25, "yd"`` -> ``"1st 25 yards"``."""
    ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(number, f"{number}th")
    return f"{ordinal} {pool_length} {_UNIT_WORD.get(unit, unit)}"


def describe_race(analysis: RaceAnalysis) -> RaceAnalysis:
    notes: list[TechniqueNote] = []
    focus: list[tuple[float, FocusArea]] = []  # (estimated seconds at stake, area)
    lengths = list(analysis.lengths)

    _race_story(analysis, lengths, notes)
    _start_feedback(analysis, notes, focus)
    _pacing_feedback(analysis, lengths, notes, focus)
    _stroke_rate_feedback(analysis, lengths, notes, focus)

    if not focus:
        stroke_lower = analysis.stroke.lower()
        focus.append((0.0, FocusArea(
            "Building more speed",
            "Nothing in this race stood out as a clear weakness, so the next step is raw speed.",
            "Swim short, fast repeats with full rest of the same stroke. Leave the wall at race speed "
            "and hold that speed to the next wall.",
            (f"{stroke_lower} sprint sets", f"{stroke_lower} speed sets with full rest"),
        )))
    focus.sort(key=lambda item: -item[0])
    return replace(
        analysis,
        summary=_summary(analysis, lengths),
        technique=tuple(notes),
        focus_areas=tuple(area for _, area in focus[:3]),
    )


def _summary(analysis: RaceAnalysis, lengths: list[LengthMetrics]) -> str:
    event = f"{analysis.distance} {analysis.unit} {analysis.stroke.lower()}"
    unit_word = _UNIT_WORD.get(analysis.unit, analysis.unit)
    parts: list[str] = []
    if analysis.race_time_s is not None:
        source = " (your official time)" if analysis.race_time_source == "official" else ""
        parts.append(f"You swam the {event} in {format_time(analysis.race_time_s)}{source}.")
        speed = analysis.distance / analysis.race_time_s
        pace = analysis.race_time_s * 100.0 / analysis.distance
        parts.append(f"That is an average of {speed:.2f} {unit_word} per second, a pace of "
                     f"{format_time(pace)} per 100 {unit_word}.")
    else:
        parts.append(f"This is your {event}. The finish was not visible, so the total time is not shown.")
    if len(lengths) == 2:
        parts.append(f"Your first {analysis.pool_length} took {format_time(lengths[0].split_s)} and your second "
                     f"{format_time(lengths[1].split_s)}.")
    elif len(lengths) > 2:
        fastest = min(lengths, key=lambda item: item.split_s)
        slowest = max(lengths, key=lambda item: item.split_s)
        parts.append(f"Your fastest length was length {fastest.number} ({format_time(fastest.split_s)}) and "
                     f"your slowest was length {slowest.number} ({format_time(slowest.split_s)}).")
    elif len(lengths) == 1 and analysis.expected_lengths > 1:
        parts.append(f"Your first {analysis.pool_length} took {format_time(lengths[0].split_s)}.")
    if analysis.average_stroke_rate_spm:
        parts.append(f"You averaged about {analysis.average_stroke_rate_spm:.0f} "
                     f"{strokes_word(analysis.stroke)} per minute.")
    return " ".join(parts)


def _race_story(analysis: RaceAnalysis, lengths: list[LengthMetrics], notes: list[TechniqueNote]) -> None:
    """A length-by-length account in ordinary swimming language."""
    if not lengths:
        return
    unit_word = _UNIT_WORD.get(analysis.unit, analysis.unit)
    pieces = []
    for length in lengths:
        label = length_label(length.number, length.distance, length.unit)
        piece = f"the {label} in {format_time(length.split_s)}"
        if length.number > 1:
            delta = length.split_s - lengths[length.number - 2].split_s
            if abs(delta) >= 0.08:
                piece += f", {abs(delta):.2f} seconds {'slower' if delta > 0 else 'faster'} than the length before"
        pieces.append(piece)
    if len(pieces) == 1:
        observation = f"You swam {pieces[0]}."
    elif len(pieces) == 2:
        observation = f"You swam {pieces[0]}, then {pieces[1]}."
    else:
        observation = "You swam " + "; ".join(pieces[:-1]) + f"; and {pieces[-1]}."
    if analysis.race_time_s is not None:
        observation += (f" Your average speed was {analysis.distance / analysis.race_time_s:.2f} "
                        f"{unit_word} per second.")

    first = lengths[0]
    if len(lengths) == 1:
        meaning = ("That is the whole race in one length, including your dive. The Race tab has the "
                   "split and the pace.")
        tone = "neutral"
    elif len(lengths) == 2:
        lost = lengths[1].split_s - first.split_s - first.split_s * _DIVE_ALLOWANCE
        if lost <= 0.04 * first.split_s:
            meaning = ("The first length includes the dive, so it should be a little faster. You held "
                       "your speed coming home.")
            tone = "strength"
        else:
            meaning = (f"The first length includes the dive, so some drop-off is normal. After that, "
                       f"you still lost about {lost:.1f} seconds on the way home.")
            tone = "work_on" if lost > 0.10 * first.split_s else "neutral"
    else:
        fastest = min(lengths, key=lambda item: item.split_s)
        slowest = max(lengths, key=lambda item: item.split_s)
        meaning = (f"Your fastest length was the {length_label(fastest.number, fastest.distance, fastest.unit)} "
                   f"and your slowest was the {length_label(slowest.number, slowest.distance, slowest.unit)}. "
                   "The first length includes the dive, so it is usually the fastest.")
        tone = "neutral"
    notes.append(TechniqueNote("The race", tone, observation, meaning))


def _start_feedback(analysis: RaceAnalysis, notes: list[TechniqueNote],
                    focus: list[tuple[float, FocusArea]]) -> None:
    reaction = analysis.reaction_s
    if reaction is None:
        return
    observation = f"You started moving {reaction:.2f} seconds after the beep."
    if reaction <= 0.30:
        notes.append(TechniqueNote("Start", "strength", observation,
                                   "That is a sharp reaction. You were moving almost as soon as the sound went."))
    elif reaction <= 0.50:
        notes.append(TechniqueNote("Start", "neutral", observation,
                                   "That is a solid reaction. The quickest starters begin moving within about "
                                   "three tenths of a second."))
    else:
        notes.append(TechniqueNote("Start", "work_on", observation,
                                   "That is slow off the beep, so you gave away time before you even left the "
                                   "block. React to the sound itself, not to the swimmers next to you."))
        focus.append((reaction - 0.35, FocusArea(
            "Reacting to the start",
            f"You took {reaction:.2f} seconds to move after the beep; quick starters move in about 0.3.",
            "Practise starts to a sound with your eyes down. Stay still and loaded so you can push "
            "the instant you hear it.",
            ("swim start reaction drills", "track start drills swimming"),
        )))


def _pacing_feedback(analysis: RaceAnalysis, lengths: list[LengthMetrics], notes: list[TechniqueNote],
                     focus: list[tuple[float, FocusArea]]) -> None:
    if len(lengths) < 2:
        return
    stroke_lower = analysis.stroke.lower()
    unit_word = _UNIT_WORD.get(analysis.unit, analysis.unit)
    first, last = lengths[0], lengths[-1]
    if len(lengths) == 2:
        loss = last.split_s - first.split_s
        observation = (f"Your second length was {abs(loss):.2f} seconds {'slower' if loss >= 0 else 'faster'} "
                       f"than your first ({format_time(first.split_s)} then {format_time(last.split_s)}). "
                       f"Your speed went from {first.average_speed:.2f} to {last.average_speed:.2f} "
                       f"{unit_word} per second.")
        lost = loss - first.split_s * _DIVE_ALLOWANCE
        if lost <= 0.04 * first.split_s:
            notes.append(TechniqueNote("Pacing", "strength", observation,
                                       "The first length includes the dive, so being a little slower on the way "
                                       "back is normal. You held your speed well."))
            return
        tone = "work_on" if lost > 0.10 * first.split_s else "neutral"
        notes.append(TechniqueNote("Pacing", tone, observation,
                                   f"Part of that gap is normal because the first length includes the dive, but "
                                   f"about {lost:.1f} seconds of it is speed you lost on the way home. That is "
                                   "the time sitting in the second half of this race."))
        focus.append((lost, FocusArea(
            "Holding your speed in the second half",
            f"You lost about {lost:.1f} seconds in your second length beyond what the dive explains.",
            "Swim broken race-pace repeats of this event and make the second length as fast as the first. "
            "You can look up sets for that below.",
            (f"{stroke_lower} race pace sets", f"back half {stroke_lower} sets",
             f"broken {analysis.distance} {stroke_lower} sets"),
        )))
        return
    middle = lengths[1:-1]
    middle_average = sum(item.split_s for item in middle) / len(middle)
    fade = last.split_s - middle_average
    half = len(lengths) // 2
    front = sum(item.split_s for item in lengths[:half])
    back = sum(item.split_s for item in lengths[half:])
    splits_text = ", ".join(format_time(item.split_s) for item in lengths)
    observation = (f"Your lengths were {splits_text}. The front half took {format_time(front)} and the back "
                   f"half {format_time(back)}.")
    spread = (max(item.split_s for item in middle) - min(item.split_s for item in middle)) / middle_average
    extra = back - front - first.split_s * _DIVE_ALLOWANCE
    if extra <= 0.04 * front and spread <= 0.06:
        notes.append(TechniqueNote("Pacing", "strength", observation,
                                   "Your lengths after the dive were very even. That is well-controlled pacing."))
        return
    meaning = (f"The back half was {back - front:.1f} seconds slower than the front half. The front half "
               "includes the dive, so some of that is normal.")
    if fade > 0.04 * middle_average:
        meaning += f" Your last length was {fade:.1f} seconds slower than your middle lengths."
    tone = "work_on" if extra > 0.08 * front or fade > 0.06 * middle_average else "neutral"
    notes.append(TechniqueNote("Pacing", tone, observation, meaning))
    if tone == "work_on":
        focus.append((max(extra, fade), FocusArea(
            "Holding your speed in the back half",
            f"Your back half was {back - front:.1f} seconds slower than your front half.",
            "Swim race-pace repeats with short rest and keep every length within a few tenths of the others. "
            "You can look up sets for that below.",
            (f"{stroke_lower} race pace sets", f"negative split {stroke_lower} sets",
             f"broken {analysis.distance} {stroke_lower} sets"),
        )))


def _stroke_rate_feedback(analysis: RaceAnalysis, lengths: list[LengthMetrics], notes: list[TechniqueNote],
                          focus: list[tuple[float, FocusArea]]) -> None:
    stroke_lower = analysis.stroke.lower()
    word = strokes_word(analysis.stroke)
    rated = [item for item in lengths if item.stroke_rate_spm]
    if len(rated) < 2:
        if analysis.average_stroke_rate_spm:
            notes.append(TechniqueNote(
                "Stroke rate", "neutral",
                f"Your stroke rate was about {analysis.average_stroke_rate_spm:.0f} {word} per minute.",
                "That is how fast your arms were turning over. Comparing it between races shows whether you "
                "got faster by moving your arms quicker or by taking a longer stroke.",
            ))
        return
    rates = ", ".join(f"{item.stroke_rate_spm:.0f}" for item in rated)
    observation = f"Your stroke rate by length was about {rates} {word} per minute."
    # Compare against the first length without a dive when there is one.
    reference = rated[1] if len(rated) >= 3 and rated[0].number == 1 else rated[0]
    last = rated[-1]
    tempo_change = (last.stroke_rate_spm - reference.stroke_rate_spm) / reference.stroke_rate_spm
    allowance = _DIVE_ALLOWANCE if reference.number == 1 else 0.0
    slowdown = last.split_s / reference.split_s - 1.0 - allowance
    if tempo_change <= -0.10:
        notes.append(TechniqueNote(
            "Stroke rate", "work_on", observation,
            f"Your arms slowed down from about {reference.stroke_rate_spm:.0f} to {last.stroke_rate_spm:.0f} "
            f"{word} per minute. When your arms slow down, you slow down with them unless each stroke "
            "starts pulling you farther. Finishing with the same turnover you had in the first half is "
            "a big part of holding your time.",
        ))
        focus.append((max(0.3, slowdown * last.split_s), FocusArea(
            "Keeping your tempo when tired",
            f"Your stroke rate fell from about {reference.stroke_rate_spm:.0f} to "
            f"{last.stroke_rate_spm:.0f} {word} per minute.",
            "Swim short repeats at your race turnover and do not let the arms slow down on the last one. "
            "A tempo trainer makes that target easy to hear. You can look up sets for that below.",
            (f"{stroke_lower} tempo trainer sets", f"{stroke_lower} stroke rate sets"),
        )))
    elif slowdown > 0.04:
        if tempo_change >= 0.10:
            meaning = (f"Your arms turned over faster at the end ({last.stroke_rate_spm:.0f} {word} per "
                       "minute), but you still slowed down. Each stroke was moving you less far: the extra "
                       "strokes were slipping through the water instead of holding on to it.")
        else:
            meaning = ("Your arms kept turning over at the same speed, but you still slowed down. That means "
                       "each stroke moved you less far late in the race — usually a shorter pull, or a hand "
                       "that slips instead of holding the water.")
        notes.append(TechniqueNote("Stroke rate", "work_on", observation, meaning))
        # Ranked just behind pacing, which reports the same lost time.
        focus.append((max(0.3, 0.8 * slowdown * last.split_s), FocusArea(
            "Getting more from each stroke",
            "Your tempo did not drop, yet you went slower at the end, so each stroke was moving you less far.",
            "Feel the water on the catch and finish every pull. Then hold that stroke while you swim fast. "
            "You can look up sets for that below.",
            (f"{stroke_lower} catch drills", f"{stroke_lower} distance per stroke drills"),
        )))
    else:
        notes.append(TechniqueNote("Stroke rate", "strength", observation,
                                   "You kept a steady tempo through the race without losing speed from it."))
