"""Small contract tests for the React-facing API and presentation layer."""

from dataclasses import replace

import pytest
from fastapi import HTTPException

from api import parse_race_time
from core.presentation import report_payload
from tests.test_coach import _report


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


def test_react_payload_hides_unmeasured_developer_details():
    payload = report_payload(_report())
    assert payload["technique"]["stroke"] == "Freestyle"
    assert payload["technique"]["title"] == "Balance, rhythm, and finish"
    assert [item["title"] for item in payload["technique"]["sections"]] == [
        "Alternating arm rhythm", "Body-line steadiness"
    ]
    assert "Kick timing" not in str(payload["technique"]["sections"])
    assert payload["time_label"] == "Official race time"
    assert payload["nutrition_tips"]


def test_each_stroke_gets_distinct_coaching_not_a_renamed_generic_report():
    report = _report()
    labels = {}
    practice_names = set()
    for stroke in ("Freestyle", "Backstroke", "Breaststroke", "Butterfly", "Individual Medley"):
        payload = report_payload(replace(report, race_label=f"100 yd {stroke} (SCY)"))
        labels[stroke] = payload["technique"]["sections"][0]["title"]
        practice_names.add(payload["training_sets"][0]["name"])
    assert len(set(labels.values())) == 5
    assert len(practice_names) == 5
