"""Proposing: a single researcher or a small team turns a seed paper into five registered experiments.

Both policies end the same way: five registrations that pass the feasibility check, held by one
conversation (the researcher's, or the team lead's) that will later make the predictions.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator

from rtaste.models import Budget, Chat, OverBudget, Reply, ask, extract_json, prompt, work
from rtaste.tools import paper_pages, paper_text, proposer_tools

PROPOSER_CAP = 60.0  # dollars for one condition: reading, search, proposing and predictions
SINGLE_TURNS = 60  # model calls a single researcher may use before it must answer
TEAM_PROPOSER_TURNS = 30  # the same, for each of a team's two proposers
TEAM_PROPOSER_SHARE = 0.35  # share of the team's cap each proposer may spend
MAX_MINUTES = 35  # the final evaluation, which runs the whole experiment again, must finish in this time
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]  # a field that must not be left blank
Number = Annotated[float, Field(allow_inf_nan=False)]
PLAIN_PATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*")


class ClosestWork(BaseModel):
    arxiv_id: Text
    difference: Text


class Event(BaseModel):
    """The registered event: a statistic, a relation and a threshold."""

    description: Text
    statistic: Text
    relation: Literal["greater", "less", "within"]
    threshold: Number
    upper_threshold: Number | None = None

    @model_validator(mode="after")
    def _interval_is_complete(self):
        if self.relation == "within" and (self.upper_threshold is None or self.upper_threshold < self.threshold):
            raise ValueError('"within" needs an upper_threshold at or above the threshold')
        return self

    def occurred(self, value: float) -> bool:
        if self.relation == "greater":
            return value > self.threshold
        if self.relation == "less":
            return value < self.threshold
        return self.threshold <= value <= self.upper_threshold


class Evaluation(BaseModel):
    procedure: Text
    raw_outputs: list[Text] = Field(min_length=1)  # plain file paths; see check_registrations
    repetitions: int = Field(1, ge=1)
    controls: Text


class Resources(BaseModel):
    needs_gpu: bool
    models: list[str]
    expected_minutes: float


class Registration(BaseModel):
    """One registered experiment (the registration form in the paper's appendix)."""

    title: Text
    seed_connection: Text
    hypothesis: Text
    contribution: Text
    closest_work: list[ClosestWork] = Field(min_length=1)
    inputs: list[Text] = Field(min_length=1)
    method: Text
    event: Event
    evaluation: Evaluation
    resources: Resources
    full_scale_study: Text


@dataclass
class Proposal:
    condition: str  # for example "opus-single" or "astra-team"
    status: str  # "done", or the reason the condition ended early
    registrations: list[dict] = field(default_factory=list)
    before_search: list[str] = field(default_factory=list)  # understanding and directions, per proposer
    drafts: list[dict] = field(default_factory=list)  # team only
    critique: str = ""  # team only
    cost: float = 0.0
    chat: Chat | None = None  # the conversation that makes the predictions next
    transcripts: dict[str, list] = field(default_factory=dict)


def check_registrations(text: str, exactly: int | None = None) -> tuple[list[dict], list[str]]:
    """The feasibility check: are the registrations complete and runnable within the limits?

    Returns the registrations that are well formed, and everything that is wrong.
    """
    try:
        items = extract_json(text)["experiments"]
    except (json.JSONDecodeError, KeyError, TypeError):
        items = None
    if not isinstance(items, list):
        return [], ['The reply is not a JSON object with an "experiments" list.']
    registrations, problems = [], []
    for number, item in enumerate(items, 1):
        try:
            registration = Registration.model_validate(item)
        except ValidationError as error:
            problems += [f"Experiment {number}: {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]
            continue
        if any(".." in path or path.startswith("seed/") or not PLAIN_PATH.fullmatch(path)
               for path in registration.evaluation.raw_outputs):
            problems.append(f"Experiment {number}: each of raw_outputs must be a plain file path inside the working "
                            "directory and outside seed/, such as outputs/scores.jsonl, with no spaces or description.")
        if not 0 < registration.resources.expected_minutes <= MAX_MINUTES:
            problems.append(f"Experiment {number}: expected_minutes must be above 0 and at most {MAX_MINUTES}, "
                            "the time allowed for the final evaluation.")
        registrations.append(registration.model_dump())
    titles = [r["title"].strip().lower() for r in registrations]
    if len(set(titles)) < len(titles):
        problems.append("Two experiments have the same title.")
    if exactly is not None and len(items) != exactly:
        problems.append(f"There must be exactly {exactly} experiments; you gave {len(items)}.")
    return registrations, problems


def _five(text: str) -> tuple[list[dict], list[str]]:
    return check_registrations(text, exactly=5)


def _not_yet(_args) -> str:
    return "Tools are not available yet. First record your understanding and the directions you are considering."


def _read_seed(chat: Chat, seed_id: str, run_tool: dict, max_turns: int) -> Reply:
    """The first step: before any search, the agent records its understanding and its directions."""
    opening = prompt("before_search", arxiv_id=seed_id, paper=paper_text(seed_id), envelope=prompt("envelope"))
    return work(chat, opening, {name: _not_yet for name in run_tool}, max_turns, images=paper_pages(seed_id))


def propose_single(model: str, seed_id: str, repo: Path, launch_day: str, cap: float = PROPOSER_CAP) -> Proposal:
    """A single researcher: one continuous context from reading the seed to five registrations."""
    proposal = Proposal(condition=f"{model}-single", status="")
    specs, run_tool = proposer_tools(repo, launch_day)
    chat = Chat(model, prompt("system") + "\n\n" + prompt("researcher"), specs, effort="max",
                purpose=f"{seed_id}/{proposal.condition}/researcher", budget=Budget(cap))
    proposal.chat = chat
    try:
        first = _read_seed(chat, seed_id, run_tool, SINGLE_TURNS)
        proposal.before_search, proposal.status = [first.text], first.status
    except OverBudget:
        proposal.status = "spending limit reached"
    if proposal.status == "done":
        task = prompt("propose", format=prompt("registration_format"))
        registrations, proposal.status = ask(chat, task, _five, run_tool, SINGLE_TURNS)
        proposal.registrations = registrations or []
    proposal.cost = chat.budget.spent
    proposal.transcripts = {"researcher": chat.transcript}
    return proposal


def propose_team(model: str, seed_id: str, repo: Path, launch_day: str, cap: float = PROPOSER_CAP) -> Proposal:
    """A research team: two proposers draft independently, a critic reviews, a lead registers five."""
    proposal = Proposal(condition=f"{model}-team", status="")
    specs, run_tool = proposer_tools(repo, launch_day)
    budget = Budget(cap)
    system = prompt("system") + "\n\n" + prompt("researcher")
    paper, pages, envelope = paper_text(seed_id), paper_pages(seed_id), prompt("envelope")

    def draft(number: int) -> tuple[str, list[dict], Chat]:
        chat = Chat(model, system, specs, effort="max", budget=budget,
                    purpose=f"{seed_id}/{proposal.condition}/proposer{number}")
        before, drafts = "", []
        try:
            reply = _read_seed(chat, seed_id, run_tool, TEAM_PROPOSER_TURNS)
            before = reply.text
            if reply.status == "done":
                reply = work(chat, prompt("draft", format=prompt("registration_format")), run_tool,
                             TEAM_PROPOSER_TURNS, spend_limit=TEAM_PROPOSER_SHARE * cap)
            if reply.status == "done":
                drafts = check_registrations(reply.text)[0][:5]  # well-formed drafts are kept, others dropped
        except OverBudget:
            pass
        return before, drafts, chat

    with ThreadPoolExecutor(2) as pool:
        proposers = list(pool.map(draft, (1, 2)))
    proposal.before_search = [before for before, _, _ in proposers]
    proposal.drafts = [d for _, drafts, _ in proposers for d in drafts]
    proposal.transcripts = {f"proposer{i}": chat.transcript for i, (_, _, chat) in enumerate(proposers, 1)}
    proposal.cost = budget.spent
    if not proposal.drafts:
        proposal.status = "the proposers returned no usable drafts"
        return proposal

    drafts_text = json.dumps(proposal.drafts, indent=1)
    critic = Chat(model, prompt("system"), effort="max", budget=budget, purpose=f"{seed_id}/{proposal.condition}/critic")
    lead = Chat(model, system, effort="max", budget=budget, purpose=f"{seed_id}/{proposal.condition}/lead")
    proposal.chat = lead
    try:
        review = work(critic, prompt("critic", arxiv_id=seed_id, paper=paper, envelope=envelope, drafts=drafts_text),
                      {}, max_turns=1, images=pages)
        proposal.critique, proposal.status = review.text, review.status
    except OverBudget:
        proposal.status = "spending limit reached"
    if proposal.status == "done":
        task = prompt("lead", arxiv_id=seed_id, paper=paper, envelope=envelope, drafts=drafts_text,
                      critique=review.text, format=prompt("registration_format"))
        registrations, proposal.status = ask(lead, task, _five, images=pages)
        proposal.registrations = registrations or []
    proposal.transcripts |= {"critic": critic.transcript, "lead": lead.transcript}
    proposal.cost = budget.spent
    return proposal
