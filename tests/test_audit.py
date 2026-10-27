"""How the two auditors' judgments are combined, without any paid call."""

import hashlib
import json

import pytest

from rtaste import audit
from rtaste.audit import (REVIEW_ANSWERS, FollowUps, Review, _answered, combined, flagged, novelty_label, obvious,
                          pool_follow_ups, quoted)
from rtaste.models import PROMPTS, Budget, parse


def judged(label):
    return {"novelty": {"label": label, "closest_work": [], "reason": ""} if label else None}


def test_the_less_novel_label_is_used():
    assert novelty_label([judged("distinct"), judged("close variant")]) == "close variant"
    assert novelty_label([judged("equivalent"), judged("unknown")]) == "equivalent"
    assert novelty_label([judged("distinct"), judged("unknown")]) == "unknown"
    assert novelty_label([judged("distinct"), judged(None)]) == "unknown"
    assert novelty_label([judged("distinct"), judged("distinct")]) == "distinct"


def audited(fidelity, validity, value):
    return {"audit": {"fidelity": fidelity, "validity": validity, "research_value": value}}


def test_one_auditor_is_enough_to_call_an_experiment_unfaithful_or_invalid():
    three = [audited("faithful", "valid", 2), audited("unfaithful", "valid", 1), audited("faithful", "invalid", 3)]
    assert combined(three) == {"fidelity": "unfaithful", "validity": "invalid", "research_value": 2}
    assert combined(three, any_flag=False) == {"fidelity": "faithful", "validity": "valid", "research_value": 2}
    assert combined(three[:2]) == {"fidelity": "unfaithful", "validity": "valid", "research_value": 1}
    missing = [audited("faithful", "valid", 2), {"audit": None}, audited("faithful", "valid", None)]
    assert combined(missing) == {"fidelity": "faithful", "validity": "valid", "research_value": 2}
    unsure = [audited("faithful", "unknown", 2), audited("unknown", "unknown", 1), audited("faithful", "valid", 3)]
    assert combined(unsure) == {"fidelity": "faithful", "validity": "unknown", "research_value": 2}


def test_an_experiment_is_obvious_if_either_auditor_finds_it_on_the_list():
    def matches(*same_as):
        return {"matches": [{"experiment": n, "same_as": s} for n, s in enumerate(same_as, 1)]}

    assert obvious([matches(None, 3, None, None, None), matches(None, None, None, 7, None)]) == [
        False, True, False, True, False]
    assert obvious([{"matches": None}, matches(1, None, None, None, None)])[0] is True
    assert obvious([{"matches": None}, {"matches": None}]) == [None] * 5


def test_follow_ups_are_pooled_in_a_fixed_order_without_model_names():
    lists = [{"model": "luna", "experiments": [{"title": "c"}]},
             {"model": "astra", "experiments": [{"title": "a"}, {"title": "b"}]}]
    assert pool_follow_ups(lists) == [{"number": 1, "title": "a"}, {"number": 2, "title": "b"},
                                      {"number": 3, "title": "c"}]


def test_an_auditor_without_a_valid_answer_is_asked_again_rather_than_saved():
    _answered("done", "x")
    _answered("refused", "x")  # a refusal is an answer of a kind and is kept
    with pytest.raises(ValueError):
        _answered("no valid answer", "x")


def test_a_list_of_follow_ups_has_exactly_ten():
    def listed(count):
        return json.dumps({"experiments": [{"title": f"t{n}", "question": "q", "method": "m"} for n in range(count)]})

    assert parse(listed(10), FollowUps)[0] and not parse(listed(10), FollowUps)[1]
    assert parse(listed(9), FollowUps)[1] and parse(listed(11), FollowUps)[1]


def test_flagged_means_unfaithful_or_invalid_and_not_low_value():
    sound = {"fidelity": "faithful", "validity": "valid", "research_value": 0}
    assert flagged(sound) is False and flagged({**sound, "validity": "invalid"}) is True
    assert flagged({**sound, "fidelity": "unfaithful"}) is True and flagged(None) is None


def reviewed(**changed):
    answers = {question: {"reason": "r", "answer": lines[0]} for question, lines in REVIEW_ANSWERS.items()}
    return json.dumps({"what_it_asks": "a", "what_came_out": "b", **answers, **changed})


def test_a_review_must_choose_one_of_the_lines_of_the_prompt_for_every_question():
    review, problems = parse(reviewed(), Review)
    assert not problems and review["relation"]["answer"] == "reruns" and review["in_the_paper"]["quote"] is None
    review, problems = parse(reviewed(obvious={"reason": "r", "answer": "rather obvious"}), Review)
    assert review is None and "obvious" in problems[0] and "most, some, few, none" in problems[0]
    assert parse(reviewed(next_step=None), Review)[1]
    review, problems = parse(reviewed(next_step={"reason": "r", "answer": "stop"}), Review)  # a stop needs its ground
    assert review is None and 'with "stop" give "what"' in problems[0]
    assert not parse(reviewed(next_step={"reason": "r", "answer": "stop", "what": "a"},
                              answerable={"reason": "r", "answer": "only if", "needs": "a base effect"},
                              prerequisite={"reason": "r", "answer": "failed", "which": "the base effect"}), Review)[1]
    assert not parse(reviewed(in_the_paper={"reason": "r", "answer": "named"}), Review)[1]  # counted as not shown, not sent back
    for question, lines in REVIEW_ANSWERS.items():  # the form and the prompt name the same lines
        assert f'"{question}": {{"reason": "one sentence", "answer": "{" | ".join(lines)}"' in PROMPTS["review"]


def test_a_copied_passage_is_looked_for_in_the_paper_without_regard_to_spacing_and_punctuation():
    paper = "We find  that the gain ( Table  4 ) is\nlargest for the smallest model, as expected."
    assert quoted("the gain (Table 4) is largest for the smallest model", paper) is True
    assert quoted("the gain is largest for the largest model here", paper) is False
    assert quoted("the gain is", paper) is False  # too short to show anything
    assert quoted(None, paper) is None and quoted("", paper) is None
    formula = "it achieves 5.1\\times and 11.0\\times speedups over the baseline"
    assert quoted("achieves 5.1× and 11.0× speedups over the baseline", formula) is True  # the sign for the command
    assert quoted("achieves 5.1\\times and 11.0\\times speedups over", formula) is True


def test_a_review_shows_the_paper_first_and_then_what_the_auditors_were_shown(monkeypatch, tmp_path):
    sent = {}

    class Chat:
        cost = 0.25

        def __init__(self, model, system, **settings):
            sent.update(settings, model=model)
            self.transcript = [{"role": "user", "text": ""}]

    def ask(chat, message, check):
        sent["message"] = message
        return check(reviewed(in_the_paper={"reason": "r", "answer": "named", "quote": "left for future work by us"}))[0], "done"

    monkeypatch.setattr(audit, "Chat", Chat)
    monkeypatch.setattr(audit, "ask", ask)
    monkeypatch.setattr(audit, "paper_text", lambda seed_id: "THE PAPER. This is left for future work by us.")
    monkeypatch.setattr(audit, "shown", lambda *run: {"notes": "NOTES", "evaluation": "EVALUATION", "files": "FILES"})
    budget = Budget(5.0)
    saved = audit.review_experiment("grok", "2600.00001", {"title": "T"}, {}, {}, tmp_path, "opus-team/3", budget)
    opening = sent["shared"]
    assert sent["budget"] is budget  # the seed's spending limit reaches the conversation
    assert saved["opening_sha256"] == hashlib.sha256(opening.encode()).hexdigest()
    assert sent["message"].startswith(opening) and "THE PAPER" in opening and "next_step" in opening
    assert "NOTES" not in opening and '"title": "T"' not in opening  # the opening is the same for a whole seed
    assert all(part in sent["message"] for part in ('"title": "T"', "NOTES", "EVALUATION", "FILES"))
    assert sent["purpose"] == "2600.00001/opus-team/3/review-grok" and sent["effort"] == "high"
    assert saved["review"]["in_the_paper"]["answer"] == "named" and saved["quotes_found"] == {
        "in_the_paper": True, "misreads_paper": None}
    assert "THE PAPER" not in saved["transcript"][0]["text"] and "NOTES" in saved["transcript"][0]["text"]

    audit.review_experiment("opus", "2600.00001", {"title": "another"}, {}, {}, tmp_path, "luna-single/1")
    assert sent["shared"] == opening  # every review of a seed opens with the same text

    monkeypatch.setattr(audit, "ask", lambda chat, message, check: (None, "refused"))
    assert audit.review_experiment("opus", "2600.00001", {}, {}, {}, tmp_path, "x")["review"] is None  # kept as it is
    monkeypatch.setattr(audit, "ask", lambda chat, message, check: (None, "spending limit reached"))
    with pytest.raises(ValueError):  # not saved, so it is asked again on the next run
        audit.review_experiment("opus", "2600.00001", {}, {}, {}, tmp_path, "x")

