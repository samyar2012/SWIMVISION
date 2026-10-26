"""AI swim coach that answers questions about one analysed race.

Questions go to a real language model: a local Ollama model first, then
OpenAI when an API key is configured. The model is given the measured facts of
the race and the conversation so far, and is told to separate what the video
measured from general coaching knowledge. When no model is reachable the
caller gets :class:`CoachUnavailable` instead of a canned answer.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

import config
from core.errors import SwimVisionError
from core.race_analysis import RaceAnalysis
from core.race_report import format_time, strokes_word

logger = logging.getLogger("swimvision.coach")

MAX_HISTORY_MESSAGES = 12
_OLLAMA_TIMEOUT_S = 180
_OPENAI_TIMEOUT_S = 90
_GEMINI_TIMEOUT_S = 90
_UNIT_WORD = {"yd": "yards", "m": "meters"}

SYSTEM_PROMPT = """You are SwimVision Coach, an experienced competitive swim coach talking with a \
swimmer (or their parent) about ONE race they filmed.

Answer the question they just asked. Do not paste a stock paragraph. Do not recycle the same reply \
if they ask something new.

You have the measured facts of this race below. Rules:
- Ground every statement about THIS race in those facts. Never invent numbers, splits, times, or \
observations that are not listed. Breathing, kick count, underwater distance, head position and \
body angles were NOT measured.
- If they ask about something the video did not measure, say so plainly, then give useful general \
advice for that stroke.
- You may use general swimming knowledge (technique cues, drills, sets, race strategy, mindset).
- Write sets the way a coach writes them on the whiteboard, e.g. "8 x 25 freestyle at race pace, \
30 seconds rest", and make sure the distances add up.
- Talk like a real coach: plain words, short paragraphs, no jargon, no numbered five-step plans, \
no words like "cycles" or "priority". Simple bullet lists and **bold** are fine.
- Keep answers under about 180 words unless they ask for more detail.
- Do not give medical advice.

RACE FACTS
{facts}"""


class CoachUnavailable(SwimVisionError):
    """No language model could be reached."""


def race_facts(report: RaceAnalysis) -> str:
    """Plain-text facts about the race, used as the model's only source of race data."""
    unit_word = _UNIT_WORD.get(report.unit, report.unit)
    lines = [f"Event: {report.distance} {report.unit} {report.stroke.lower()} ({report.pool_length} "
             f"{report.unit} pool, {report.expected_lengths} lengths)."]
    if report.race_time_s is not None:
        source = {"official": "official time entered by the swimmer",
                  "video": "timed from the video"}.get(report.race_time_source, "")
        lines.append(f"Race time: {format_time(report.race_time_s)} ({source}).")
        lines.append(f"Average speed: {report.distance / report.race_time_s:.2f} {unit_word} per second; pace "
                     f"{format_time(report.race_time_s * 100 / report.distance)} per 100 {unit_word}.")
        if report.race_time_source == "official" and report.video_race_time_s is not None:
            lines.append(f"The video itself measured {format_time(report.video_race_time_s)}.")
    else:
        lines.append("Race time: not available (the finish was not visible and no official time was entered).")
    if report.reaction_s is not None:
        lines.append(f"Reaction time: {report.reaction_s:.2f} s from the start beep to first movement.")
    else:
        lines.append("Reaction time: not measured (no start beep audible).")
    if report.lengths:
        lines.append("Splits by length:")
        for length in report.lengths:
            rate = (f", stroke rate about {length.stroke_rate_spm:.0f} {strokes_word(report.stroke)}/min"
                    if length.stroke_rate_spm else "")
            lines.append(f"  Length {length.number}: {format_time(length.split_s)} (total "
                         f"{format_time(length.cumulative_s)}), {length.average_speed:.2f} {unit_word}/s, pace "
                         f"{format_time(length.pace_per_100_s)} per 100{rate}.")
        lines.append("Length 1 includes the dive start, so it is normally the fastest.")
    else:
        lines.append("Splits by length: not available (not every wall was visible).")
    if report.average_stroke_rate_spm:
        lines.append(f"Average stroke rate: about {report.average_stroke_rate_spm:.0f} "
                     f"{strokes_word(report.stroke)} per minute.")
    if report.technique:
        lines.append("What the analysis found:")
        lines.extend(f"  {note.topic}: {note.observation} {note.meaning}" for note in report.technique)
    if report.focus_areas:
        lines.append("What to work on (most important first):")
        lines.extend(f"  {area.title}: {area.why}" for area in report.focus_areas)
    if report.notes:
        lines.append("Notes: " + " ".join(report.notes))
    lines.append("Not measured by this video: breathing, kick, underwater phases, head and body position.")
    return "\n".join(lines)


def _clean_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    messages = []
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        role, content = item.get("role"), (item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:2000]})
    return messages


def ask_coach(report: RaceAnalysis, question: str, history: list[dict[str, str]] | None = None) -> str:
    """Answer ``question`` about the race with a language model."""
    question = question.strip()
    if not question:
        raise SwimVisionError("Type a question for the coach.")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(facts=race_facts(report))},
        *_clean_history(history),
        {"role": "user", "content": question},
    ]
    errors: list[str] = []
    # Cloud models first: a local 7B chat model answers too fast and too vaguely
    # to sound like a coach. Ollama remains the offline fallback.
    if config.OPENAI_API_KEY:
        try:
            return _ask_openai(messages)
        except CoachUnavailable as exc:
            errors.append(str(exc))
    if config.GEMINI_API_KEY:
        try:
            return _ask_gemini(messages)
        except CoachUnavailable as exc:
            errors.append(str(exc))
    try:
        return _ask_ollama(messages)
    except CoachUnavailable as exc:
        errors.append(str(exc))
    logger.warning("AI coach unavailable: %s", "; ".join(errors))
    raise CoachUnavailable(
        "The AI coach is offline. Add an OPENAI_API_KEY or GEMINI_API_KEY to .env, "
        f"or start Ollama (ollama serve) with the {config.OLLAMA_MODEL} model, then ask again."
    )


def _ollama_model() -> str:
    """The configured model if installed, otherwise the first installed chat model."""
    response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
    response.raise_for_status()
    installed = [item["name"] for item in response.json().get("models", [])]
    if not installed:
        raise CoachUnavailable("Ollama is running but has no models installed.")
    if config.OLLAMA_MODEL in installed:
        return config.OLLAMA_MODEL
    chat = [name for name in installed if "embed" not in name]
    return (chat or installed)[0]


def _ask_ollama(messages: list[dict[str, str]]) -> str:
    try:
        model = _ollama_model()
        response = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"temperature": 0.5, "num_ctx": 4096}},
            timeout=_OLLAMA_TIMEOUT_S,
        )
        response.raise_for_status()
        answer = _content(response.json().get("message"))
    except (requests.RequestException, ValueError) as exc:
        raise CoachUnavailable(f"Ollama: {exc}") from exc
    if not answer:
        raise CoachUnavailable("Ollama returned an empty answer.")
    return answer


def _ask_openai(messages: list[dict[str, str]]) -> str:
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"model": config.OPENAI_COACH_MODEL, "messages": messages,
                  "temperature": 0.6, "max_tokens": 700},
            timeout=_OPENAI_TIMEOUT_S,
        )
        response.raise_for_status()
        choices = response.json().get("choices") or [{}]
        answer = _content(choices[0].get("message"))
    except (requests.RequestException, ValueError) as exc:
        raise CoachUnavailable(f"OpenAI: {exc}") from exc
    if not answer:
        raise CoachUnavailable("OpenAI returned an empty answer.")
    return answer


def _ask_gemini(messages: list[dict[str, str]]) -> str:
    system = next((item["content"] for item in messages if item["role"] == "system"), "")
    contents = []
    for item in messages:
        if item["role"] == "system":
            continue
        role = "model" if item["role"] == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n\n" + item["content"]
        else:
            contents.append({"role": role, "parts": [{"text": item["content"]}]})
    if not contents or contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": "Please answer about this race."}]})
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_COACH_MODEL}:generateContent",
            headers={"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": contents,
                "generationConfig": {"temperature": 0.6, "maxOutputTokens": 700},
            },
            timeout=_GEMINI_TIMEOUT_S,
        )
        response.raise_for_status()
        parts = (((response.json().get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
        answer = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    except (requests.RequestException, ValueError, TypeError) as exc:
        raise CoachUnavailable(f"Gemini: {exc}") from exc
    if not answer:
        raise CoachUnavailable("Gemini returned an empty answer.")
    return answer


def _content(message: Any) -> str:
    return (message or {}).get("content", "").strip() if isinstance(message, dict) else ""
