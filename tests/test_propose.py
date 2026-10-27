"""The feasibility check, without any paid call."""

import json

from rtaste.propose import Event, check_registrations

GOOD = {
    "title": "t", "seed_connection": "s", "hypothesis": "h", "contribution": "c",
    "closest_work": [{"arxiv_id": "2501.00001", "difference": "d"}], "inputs": ["x"], "method": "m",
    "event": {"description": "e", "statistic": "accuracy gap", "relation": "greater", "threshold": 0.02},
    "evaluation": {"procedure": "p", "raw_outputs": ["out.jsonl"], "repetitions": 1, "controls": "none needed"},
    "resources": {"needs_gpu": True, "models": ["Qwen3-4B"], "expected_minutes": 30},
    "full_scale_study": "f",
}


def reply(experiments) -> str:
    return "Here they are:\n```json\n" + json.dumps({"experiments": experiments}) + "\n```"


def five(**changes):
    return [{**GOOD, "title": f"t{i}", **changes} for i in range(5)]


def test_five_complete_registrations_pass():
    registrations, problems = check_registrations(reply(five()), exactly=5)
    assert len(registrations) == 5 and problems == []


def test_missing_field_wrong_count_and_long_run_are_reported():
    broken = five()
    del broken[0]["method"]
    broken[1]["resources"] = {**GOOD["resources"], "expected_minutes": 300}
    _, problems = check_registrations(reply(broken[:4]), exactly=5)
    text = " ".join(problems)
    assert "Experiment 1: method" in text and "at most 35" in text and "exactly 5" in text


def test_raw_outputs_must_be_plain_file_paths():
    described = {**GOOD["evaluation"], "raw_outputs": ["scores.jsonl with one row per item", "../outside.json"]}
    kept, problems = check_registrations(reply(five(evaluation=described)), exactly=5)
    assert sum("plain file path" in p for p in problems) == 5
    assert len(kept) == 5  # reported but kept, so a team proposer's drafts are not lost over it


def test_reply_without_json_is_rejected():
    assert check_registrations("I could not finish.", exactly=5)[1]
    assert check_registrations('{"experiments": null}', exactly=5)[1]


def test_empty_fields_are_reported():
    _, problems = check_registrations(reply(five(hypothesis="", closest_work=[], inputs=[])), exactly=5)
    text = " ".join(problems)
    assert "Experiment 1: hypothesis" in text and "closest_work" in text and "inputs" in text


def test_blank_text_an_endless_threshold_and_outputs_inside_the_seed_code_are_reported():
    endless = {**GOOD["event"], "threshold": float("inf")}
    _, problems = check_registrations(reply(five(method="   ", event=endless)), exactly=5)
    assert "Experiment 1: method" in " ".join(problems) and "event.threshold" in " ".join(problems)
    inside = {**GOOD["evaluation"], "raw_outputs": ["seed/results.json"]}
    assert sum("outside seed/" in p for p in check_registrations(reply(five(evaluation=inside)), exactly=5)[1]) == 5


def test_event_relations():
    assert Event(description="e", statistic="s", relation="greater", threshold=1).occurred(1.5)
    assert not Event(description="e", statistic="s", relation="less", threshold=1).occurred(1.5)
    assert Event(description="e", statistic="s", relation="within", threshold=1, upper_threshold=2).occurred(1.5)
