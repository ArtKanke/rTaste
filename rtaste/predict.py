"""Predictions and the two commitments that come before any experiment is run.

First the five registrations are committed. Only then does the agent give, for each experiment,
the probability that its registered event occurs, and rank the five. Fresh judges, who see the
seed and the committed experiments but nothing of the agent's reasoning, give the same
predictions. All of these are committed in turn, and execution starts after that.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, model_validator

from rtaste.models import Chat, ask, now, parse, prompt
from rtaste.propose import Proposal
from rtaste.tools import paper_pages, paper_text

JUDGES = ("astra", "fable", "opus", "sol", "luna", "muse")  # a fresh context of each proposing model


class Prediction(BaseModel):
    experiment: int
    probability: float = Field(ge=0, le=1)
    reason: str = ""


class Predictions(BaseModel):
    """One forecaster's predictions for the five experiments of a condition."""

    predictions: list[Prediction]
    ranking: list[int]  # experiment numbers, the most worth pursuing first
    ranking_reason: str = ""

    @model_validator(mode="after")
    def _covers_each_experiment_once(self):
        if sorted(p.experiment for p in self.predictions) != [1, 2, 3, 4, 5]:
            raise ValueError("predictions must cover experiments 1 to 5, each once")
        if sorted(self.ranking) != [1, 2, 3, 4, 5]:
            raise ValueError("ranking must list experiments 1 to 5, each once")
        self.predictions.sort(key=lambda p: p.experiment)
        return self


def content_hash(content) -> str:
    """SHA-256 of the content written as JSON with sorted keys and no extra spaces."""
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def commit(content: dict) -> dict:
    """Fix the content: store it with its hash and the time, so any later change can be detected."""
    return {"content": content, "sha256": content_hash(content),
            "committed_at": now()}


def intact(commitment: dict) -> bool:
    return content_hash(commitment["content"]) == commitment["sha256"]


def shuffled(experiments: list[dict], reader: str) -> list[dict]:
    """The numbered experiments in an order that differs from reader to reader but is always the same
    for one reader. An agent tends to register its favourite first, so a reader shown the registered
    order could rank well by position alone. The reader is named by seed, condition, role and model,
    so a model that both judges and ranks sees two orders, and the orders differ from seed to seed."""
    return sorted(experiments, key=lambda e: content_hash(f"{reader}/{e['experiment']}"))


def first_commitment(seed_id: str, proposal: Proposal) -> dict:
    """Commit the five registrations (which name their inputs) together with the execution limits."""
    return commit({"seed": seed_id, "condition": proposal.condition,
                   "registrations": proposal.registrations, "limits": prompt("envelope")})


def second_commitment(first: dict, agent: dict, judges: list[dict]) -> dict:
    """Commit the predictions of the agent and of the fresh judges, tied to the committed experiments.

    A forecaster that gave no valid predictions is committed as None.
    """
    return commit({"first_commitment": first["sha256"], "agent": agent["predictions"],
                   "judges": {judge["judge"]: judge["predictions"] for judge in judges}})


def _closed(_args) -> str:
    return "Tools are no longer available. The experiments are committed; give your predictions."


def _check(text: str) -> tuple[dict | None, list[str]]:
    return parse(text, Predictions)


def predict_agent(proposal: Proposal) -> dict:
    """The agent's own predictions, made in the context that proposed the experiments."""
    chat = proposal.chat
    titles = "\n".join(f"{n}. {r['title']}" for n, r in enumerate(proposal.registrations, 1))
    spent_before, said_before = chat.cost, len(chat.transcript)
    predictions, status = ask(chat, prompt("predict", titles=titles, format=prompt("prediction_format")), _check,
                              {tool["name"]: _closed for tool in chat.tools}, max_turns=chat.turns + 4)
    return {"status": status, "predictions": predictions, "cost": chat.cost - spent_before,
            "transcript": chat.transcript[said_before:]}


def predict_fresh(model: str, seed_id: str, registrations: list[dict], condition: str) -> dict:
    """A fresh judge's predictions. It sees the seed, the five experiments and the limits, and has no tools."""
    chat = Chat(model, prompt("system"), effort="max", purpose=f"{seed_id}/{condition}/judge-{model}")
    numbered = shuffled([{"experiment": n, **r} for n, r in enumerate(registrations, 1)], chat.purpose)
    predictions, status = ask(chat, prompt(
        "judge", arxiv_id=seed_id, paper=paper_text(seed_id), envelope=prompt("envelope"),
        experiments=json.dumps(numbered, indent=1), format=prompt("prediction_format")), _check,
        images=paper_pages(seed_id))
    return {"judge": model, "status": status, "predictions": predictions, "cost": chat.cost,
            "transcript": chat.transcript}


def predict_all_fresh(seed_id: str, registrations: list[dict], condition: str, judges=JUDGES) -> list[dict]:
    """All fresh judges for one condition, at the same time."""
    with ThreadPoolExecutor(len(judges)) as pool:
        return list(pool.map(lambda model: predict_fresh(model, seed_id, registrations, condition), judges))
