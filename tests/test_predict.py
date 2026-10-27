"""Predictions and commitments, without any paid call."""

import json

import pytest
from pydantic import ValidationError

from rtaste.models import Reply, ask
from rtaste.predict import Predictions, _check, commit, intact, second_commitment, shuffled


def answer(probabilities=(0.1, 0.2, 0.3, 0.4, 0.5), ranking=(5, 4, 3, 2, 1)) -> dict:
    return {"predictions": [{"experiment": n, "probability": p, "reason": "r"}
                            for n, p in enumerate(probabilities, 1)],
            "ranking": list(ranking), "ranking_reason": "r"}


class FakeChat:
    """Stands in for a conversation and gives prepared replies."""

    tools, turns, cost = [], 0, 0.0
    budget = None

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def send(self, text, images=()):
        self.sent.append(text)
        return self.replies.pop(0)


def test_valid_predictions_are_accepted_in_experiment_order():
    shuffled = answer()
    shuffled["predictions"].reverse()
    assert [p.experiment for p in Predictions.model_validate(shuffled).predictions] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("broken", [
    answer(probabilities=(0.1, 0.2, 0.3, 0.4, 1.5)),  # not a probability
    answer(probabilities=(0.1, 0.2, 0.3, 0.4)),  # one experiment missing
    answer(ranking=(1, 2, 3, 4, 4)),  # not a ranking of the five
])
def test_invalid_predictions_are_rejected(broken):
    with pytest.raises(ValidationError):
        Predictions.model_validate(broken)


def test_an_invalid_reply_is_sent_back_and_the_correction_accepted():
    chat = FakeChat([Reply("no JSON here", [], "done", 0.0), Reply(json.dumps(answer()), [], "done", 0.0)])
    predictions, status = ask(chat, "predict", _check)
    assert status == "done" and predictions["ranking"] == [5, 4, 3, 2, 1]
    assert "cannot be accepted" in chat.sent[1]


def test_three_invalid_replies_or_a_refusal_give_no_predictions():
    assert ask(FakeChat([Reply("{}", [], "done", 0.0)] * 3), "predict", _check) == (None, "no valid answer")
    assert ask(FakeChat([Reply("", [], "refused", 0.0)]), "predict", _check) == (None, "refused")


def test_a_commitment_detects_any_later_change():
    first = commit({"registrations": [{"title": "a"}], "limits": "x"})
    assert intact(first)
    second = second_commitment(first, {"predictions": answer()}, [{"judge": "luna", "predictions": None}])
    assert second["content"]["first_commitment"] == first["sha256"] and intact(second)
    second["content"]["agent"]["predictions"][0]["probability"] = 0.9
    assert not intact(second)


def test_each_reader_gets_its_own_fixed_order_of_the_five_experiments():
    five = [{"experiment": n} for n in range(1, 6)]
    orders = {reader: [e["experiment"] for e in shuffled(five, reader)] for reader in ("a-single/opus", "a-single/sol", "b-team/opus")}
    assert all(sorted(order) == [1, 2, 3, 4, 5] for order in orders.values())
    assert len({tuple(order) for order in orders.values()}) > 1
    assert orders["a-single/opus"] == [e["experiment"] for e in shuffled(five, "a-single/opus")]
