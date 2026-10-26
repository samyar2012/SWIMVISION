"""Contract tests for the React-facing API and presentation layer."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from api import app, parse_race_time
from core.coach import CoachUnavailable
from core.geometry import BBox
from core.models import RaceInfo
from core.presentation import report_payload
from core.video import VideoMetadata
from storage.analyses import AnalysisRecord
from tests.test_race_analysis import _analysis


@pytest.mark.parametrize(
    ("entered", "expected"),
    [("54.32", 54.32), ("1:54.32", 114.32), (" 2:03 ", 123.0), (None, None), ("", None)],
)
def test_official_time_parser(entered, expected):
    assert parse_race_time(entered) == expected


@pytest.mark.parametrize("entered", ["0", "1:72", "1:2:3", "race time"])
def test_official_time_parser_rejects_unclear_values(entered):
    with pytest.raises(HTTPException) as error:
        parse_race_time(entered)
    assert error.value.status_code == 422


def test_payload_has_real_splits_with_pace_and_stroke_rate():
    payload = report_payload(_analysis([14.42, 16.58], [113.0, 115.0]))

    assert payload["race_time"] == "31.00" and payload["race_time_source"] == "video"
    assert payload["complete"] is True
    assert [row["split"] for row in payload["splits"]] == ["14.42", "16.58"]
    assert [row["cumulative"] for row in payload["splits"]] == ["14.42", "31.00"]
    assert [row["label"] for row in payload["splits"]] == ["1st 25 yards", "2nd 25 yards"]
    assert [row["distance_label"] for row in payload["splits"]] == ["25 yd", "50 yd"]
    assert [row["delta_s"] for row in payload["splits"]] == [None, 2.16]
    assert [row["pace_per_100"] for row in payload["splits"]] == ["57.68", "1:06.32"]
    assert [row["stroke_rate"] for row in payload["splits"]] == ["113", "115"]
    assert payload["fastest_length"] == 1
    assert payload["stroke_rate_unit"] == "arm strokes per minute"
    assert {item["label"] for item in payload["highlights"]} == {
        "Race time", "Pace per 100 yd", "Reaction", "Stroke rate",
    }


def test_payload_technique_and_focus_are_plain_language():
    payload = report_payload(_analysis([14.5, 17.5], [56.0, 54.0], reaction=0.7, stroke="Butterfly"))
    text = str(payload["overview"]) + str(payload["technique"]) + str(payload["focus"])

    assert payload["technique"] and payload["focus"]
    assert set(payload["technique"][0]) == {"topic", "tone", "observation", "meaning"}
    for area in payload["focus"]:
        assert set(area) == {"title", "why", "how", "look_up"}
        assert area["look_up"] and all(isinstance(item, str) for item in area["look_up"])
    assert not re.search(r"\b(cycles?|priority|working well|confidence|practice plan)\b", text, re.IGNORECASE)
    for removed in ("training_sets", "nutrition_tips", "practice", "time_label"):
        assert removed not in payload
    assert payload["suggested_questions"]


def test_payload_without_splits_or_finish_says_so_without_guessing():
    report = replace(_analysis([14.0], [112.0]), status="partial", race_time_s=None, race_time_source="unknown")

    payload = report_payload(report)

    assert payload["race_time"] is None and payload["complete"] is False
    assert "Race time" not in {item["label"] for item in payload["highlights"]}
    assert payload["fastest_length"] is None


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
@pytest.fixture
def client_with_record(monkeypatch, tmp_path):
    video = tmp_path / "tracked.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2048)
    record = AnalysisRecord(
        "abc123", "race.mp4", "uploads/race.mp4", RaceInfo("Freestyle", 50, "SCY"),
        VideoMetadata(fps=30.0, frame_count=900, duration_s=30.0, width=848, height=464, codec="avc1"),
        performance=_analysis([14.42, 16.58], [113.0, 115.0]), annotated_video=str(video),
    )
    saved: list[AnalysisRecord] = []
    monkeypatch.setattr(api, "load_analysis", lambda analysis_id: record if analysis_id == "abc123"
                        else (_ for _ in ()).throw(FileNotFoundError(analysis_id)))
    monkeypatch.setattr(api, "save_analysis", saved.append)
    return TestClient(app), record, saved


def test_report_endpoint_returns_the_payload_and_video_flag(client_with_record):
    client, _, _ = client_with_record

    response = client.get("/api/analyses/abc123")

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"] == "abc123" and body["has_video"] is True
    assert [row["split"] for row in body["splits"]] == ["14.42", "16.58"]
    assert client.get("/api/analyses/missing").status_code == 404


def test_highlighted_video_supports_seeking(client_with_record):
    client, _, _ = client_with_record

    full = client.get("/api/analyses/abc123/video")
    part = client.get("/api/analyses/abc123/video", headers={"Range": "bytes=0-99"})

    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4"
    assert part.status_code == 206 and len(part.content) == 100


def test_live_view_serves_the_latest_frame_only_while_running(client_with_record):
    client, _, _ = client_with_record
    assert client.get("/api/analyses/abc123/live").status_code == 404

    api._live_frames["abc123"] = b"\xff\xd8jpeg"
    try:
        response = client.get("/api/analyses/abc123/live")
    finally:
        api._live_frames.pop("abc123", None)

    assert response.status_code == 200 and response.content == b"\xff\xd8jpeg"
    assert response.headers["cache-control"] == "no-store"


@pytest.fixture
def opening_frame(monkeypatch):
    monkeypatch.setattr(api, "read_frame", lambda *_args: np.zeros((464, 848, 3), np.uint8))


def test_tap_finds_a_swimmer_the_full_frame_missed_instead_of_a_nearby_official(
    client_with_record, monkeypatch, opening_frame,
):
    client, _, saved = client_with_record
    official = BBox(300, 120, 420, 330)   # seated official just left of the tap
    swimmer = BBox(415, 170, 478, 289)    # only detected in the enlarged crop
    monkeypatch.setattr(api, "_people", lambda _image, _x0, _y0, zoom: [official] if zoom == 1.0 else [swimmer])

    response = client.post("/api/analyses/abc123/selection", json={"x": 445 / 848, "y": 200 / 464})

    assert response.status_code == 200
    box = response.json()["box"]
    assert box["x1"] == pytest.approx(415 / 848) and box["y2"] == pytest.approx(289 / 464)
    assert saved[-1].selections[0].box == swimmer


def test_tap_prefers_the_person_whose_body_is_centred_on_it(client_with_record, monkeypatch, opening_frame):
    client, _, saved = client_with_record
    behind = BBox(200, 60, 520, 420)     # large box whose edge reaches the tap
    swimmer = BBox(415, 170, 478, 289)
    monkeypatch.setattr(api, "_people", lambda *_args: [behind, swimmer])

    client.post("/api/analyses/abc123/selection", json={"x": 445 / 848, "y": 230 / 464})

    assert saved[-1].selections[0].box == swimmer


def test_tap_on_nobody_uses_a_box_around_the_tap(client_with_record, monkeypatch, opening_frame):
    client, _, saved = client_with_record
    far_official = BBox(100, 120, 200, 330)
    monkeypatch.setattr(api, "_people", lambda _image, _x0, _y0, zoom: [far_official] if zoom == 1.0 else [])

    client.post("/api/analyses/abc123/selection", json={"x": 0.5, "y": 0.5})

    box = saved[-1].selections[0].box
    assert box.x1 < 424 < box.x2 and box.y1 < 232 < box.y2
    assert box != far_official


def test_coach_endpoint_passes_history_and_reports_an_offline_model(client_with_record, monkeypatch):
    client, _, _ = client_with_record
    seen = {}

    def fake_coach(report, question, history):
        seen.update(question=question, history=history)
        return "Hold your stroke longer on the way home."

    monkeypatch.setattr(api, "ask_coach", fake_coach)
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    response = client.post("/api/analyses/abc123/coach", json={"question": "Why?", "history": history})

    assert response.status_code == 200
    assert response.json() == {"answer": "Hold your stroke longer on the way home."}
    assert seen == {"question": "Why?", "history": history}

    def offline(*_args):
        raise CoachUnavailable("The AI coach is offline.")

    monkeypatch.setattr(api, "ask_coach", offline)
    response = client.post("/api/analyses/abc123/coach", json={"question": "Why?"})
    assert response.status_code == 503 and "offline" in response.json()["detail"]

    bad_role = client.post("/api/analyses/abc123/coach",
                           json={"question": "Why?", "history": [{"role": "system", "content": "x"}]})
    assert bad_role.status_code == 422


@pytest.mark.parametrize("stroke", ["Individual Medley", "IM"])
def test_individual_medley_is_not_accepted(client_with_record, stroke):
    client, _, _ = client_with_record

    response = client.post(
        "/api/analyses", data={"stroke": stroke, "distance": "100", "pool": "SCY"},
        files={"video": ("race.mp4", b"not a video", "video/mp4")},
    )

    assert response.status_code == 422
