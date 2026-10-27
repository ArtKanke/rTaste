"""Judgments by the auditor models.

Before execution: how novel each committed experiment was on the launch day, and whether it is an
obvious follow up, one that a model lists after reading the seed alone. After execution: the
fidelity, validity and research value of each delivered experiment, and a ranking of each
condition's five experiments by research value. Added after the study: a review of each delivered
experiment, of its idea and of what came out, by the same three models.
"""

import copy as copy_module
import hashlib
import io
import json
import re
import tarfile
import unicodedata
import urllib.error
from pathlib import Path
from string import Template
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from rtaste.models import PROMPTS, Budget, Chat, ask, parse, prompt
from rtaste.predict import JUDGES, shuffled
from rtaste.propose import ClosestWork
from rtaste.tools import arxiv_tools, paper_info, paper_pages, paper_text

AUDITORS = ("opus", "sol", "grok")  # the two models that execute, and one that neither proposes nor executes
NOVELTY_JUDGES = ("opus", "sol")  # they judge novelty and obvious follow ups, before execution
NOVELTY_TURNS = 12  # model calls one novelty judgment may use
NOVELTY_SPEND = 0.60  # dollars after which a novelty judgment must answer
LEAST_NOVEL_FIRST = ("equivalent", "close variant", "unknown", "distinct")
LOWEST_FIRST = {"fidelity": ("unfaithful", "unknown", "faithful"), "validity": ("invalid", "unknown", "valid")}
CODE = (".py", ".sh", ".ipynb", ".yaml", ".yml", ".toml", ".cfg")
EVIDENCE_CHARS = 240_000  # about 60,000 tokens of files shown to an auditor
CODE_FILE_CHARS, OTHER_FILE_CHARS = 40_000, 6_000
LARGEST_READ = 20_000_000  # bytes; a larger file in an archive is named to the auditor but not opened


class Novelty(BaseModel):
    label: Literal["equivalent", "close variant", "distinct", "unknown"]
    closest_work: list[ClosestWork]
    reason: str


class FollowUp(BaseModel):
    title: str
    question: str
    method: str


class FollowUps(BaseModel):
    experiments: list[FollowUp] = Field(min_length=10, max_length=10)


class Match(BaseModel):
    experiment: int
    same_as: int | None
    reason: str = ""


class Matches(BaseModel):
    matches: list[Match]

    @model_validator(mode="after")
    def _covers_each_experiment_once(self):
        if sorted(m.experiment for m in self.matches) != [1, 2, 3, 4, 5]:
            raise ValueError("matches must cover experiments 1 to 5, each once")
        self.matches.sort(key=lambda m: m.experiment)
        return self


def judge_novelty(auditor: str, seed_id: str, registration: dict, launch_day: str, name: str) -> dict:
    """One auditor's novelty judgment of one committed experiment, made with the arXiv tools."""
    seed = paper_info(seed_id)
    specs, run_tool = arxiv_tools(launch_day)

    def check(text: str) -> tuple[dict | None, list[str]]:
        novelty, problems = parse(text, Novelty)
        if novelty and novelty["label"] == "equivalent":
            try:
                cited = paper_info(novelty["closest_work"][0]["arxiv_id"]) if novelty["closest_work"] else None
            except urllib.error.HTTPError as error:
                if error.code != 400:  # 400 is arXiv's answer to something that is not an arXiv id
                    raise
                cited = None
            if cited is None or cited["posted"] > launch_day:
                problems = ['An "equivalent" judgment must name, first in closest_work, an arXiv paper '
                            "posted on or before the launch day."]
        return novelty, problems

    chat = Chat(auditor, prompt("system"), specs, effort="high", purpose=f"{seed_id}/{name}/novelty-{auditor}")
    novelty, status = ask(chat, prompt(
        "novelty", title=seed["title"], arxiv_id=seed_id, abstract=seed["abstract"], launch_day=launch_day,
        experiment=json.dumps(registration, indent=1)), check, run_tool, NOVELTY_TURNS, NOVELTY_SPEND)
    return {"auditor": auditor, "status": status, "novelty": novelty, "cost": chat.cost, "transcript": chat.transcript}


def novelty_label(judgments: list[dict]) -> str:
    """The label used when the two judges disagree: the less novel one. No judgment counts as unknown."""
    labels = [j["novelty"]["label"] if j and j["novelty"] else "unknown" for j in judgments]
    return min(labels, key=LEAST_NOVEL_FIRST.index)


def list_follow_ups(model: str, seed_id: str) -> dict:
    """The follow-up experiments one model would propose from the seed alone, without search."""
    chat = Chat(model, prompt("system"), effort="max", purpose=f"{seed_id}/follow-ups-{model}")
    answer, status = ask(chat, prompt("follow_ups", arxiv_id=seed_id, paper=paper_text(seed_id),
                                      envelope=prompt("envelope")), lambda text: parse(text, FollowUps),
                         images=paper_pages(seed_id))
    _answered(status, f"follow ups by {model}")
    return {"model": model, "status": status, "experiments": answer["experiments"] if answer else [],
            "cost": chat.cost}


def pool_follow_ups(lists: list[dict]) -> list[dict]:
    """All models' follow ups in one numbered list, without the names of the models."""
    pooled = [experiment for one in sorted(lists, key=lambda one: JUDGES.index(one["model"]))
              for experiment in one["experiments"]]
    return [{"number": n, **experiment} for n, experiment in enumerate(pooled, 1)]


def match_follow_ups(auditor: str, seed_id: str, registrations: list[dict], follow_ups: list[dict],
                     condition: str) -> dict:
    """One auditor's judgment of which of a condition's five experiments are on the list of follow ups."""
    numbered = [{"experiment": n, **r} for n, r in enumerate(registrations, 1)]

    def check(text: str) -> tuple[dict | None, list[str]]:
        answer, problems = parse(text, Matches)
        if answer and any(m["same_as"] is not None and not 1 <= m["same_as"] <= len(follow_ups)
                          for m in answer["matches"]):
            problems = [f"same_as must be a number from 1 to {len(follow_ups)}, or null."]
        return answer, problems

    chat = Chat(auditor, prompt("system"), effort="high", purpose=f"{seed_id}/{condition}/follow-ups-{auditor}")
    answer, status = ask(chat, prompt("match_follow_ups", arxiv_id=seed_id, follow_ups=json.dumps(follow_ups, indent=1),
                                      experiments=json.dumps(numbered, indent=1)), check)
    return {"auditor": auditor, "status": status, "matches": answer["matches"] if answer else None, "cost": chat.cost,
            "transcript": chat.transcript}


def obvious(judgments: list[dict]) -> list[bool | None]:
    """For each of the five experiments: did either auditor find it on the list? None if neither judged."""
    judged = [j["matches"] for j in judgments if j["matches"]]
    if not judged:
        return [None] * 5
    return [any(matches[n]["same_as"] is not None for matches in judged) for n in range(5)]


class Audit(BaseModel):
    departures: list[str]  # every difference between the registration and what was run
    fidelity: Literal["faithful", "unfaithful", "unknown"]
    validity: Literal["valid", "invalid", "unknown"]
    research_value: Literal[0, 1, 2, 3] | None
    problems: list[str]
    reason: str


class ValueRanking(BaseModel):
    ranking: list[int]
    reason: str = ""

    @model_validator(mode="after")
    def _ranks_each_experiment_once(self):
        if sorted(self.ranking) != [1, 2, 3, 4, 5]:
            raise ValueError("ranking must list experiments 1 to 5, each once")
        return self


def evidence(archive: Path, written: list[str] | None = None) -> str:
    """The files the protected evaluation left, as text for an auditor: code in full first, then the
    results, outputs and logs, each cut to a length. Anything left out is named at the end.

    `written` lists the files the evaluation itself wrote. The others came with the submission, for
    example logs of the engineer's own earlier runs, and are marked so that they are not mistaken
    for results of the evaluation."""
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        environments = [m.name.removesuffix("pyvenv.cfg") for m in members if m.name.endswith("pyvenv.cfg")]
        members = [m for m in members
                   if not any(m.name.startswith(env) for env in environments) and "__pycache__/" not in m.name]
        large = [f"{m.name.removeprefix('./')} ({m.size} bytes)" for m in members if m.size > LARGEST_READ]
        files = [(m.name.removeprefix("./"), tar.extractfile(m).read()) for m in members if m.size <= LARGEST_READ]
    # Files near the top come first, then code before the rest: a cache several folders deep can hold
    # thousands of generated files, and it must not take the place of the experiment's own outputs.
    files.sort(key=lambda f: (f[0].count("/") > 2, not f[0].endswith(CODE), f[0] != "result.json", f[0]))
    listed = sorted((m.name.removeprefix("./") for m in members), key=lambda name: (name.count("/"), name))
    sizes = {m.name.removeprefix("./"): m.size for m in members}
    listing = "\n".join(f"{name} ({sizes[name]} bytes)" for name in listed[:400])
    listing += f"\n... and {len(listed) - 400} more files" if len(listed) > 400 else ""
    parts, left_out, used = [f"=== files in the archive ===\n{listing}"], large, 0
    for name, data in files:
        if b"\0" in data[:2000]:
            left_out.append(f"{name} (binary, {len(data)} bytes)")
            continue
        text = data.decode("utf-8", "replace")
        limit = CODE_FILE_CHARS if name.endswith(CODE) else OTHER_FILE_CHARS
        if len(text) > limit:
            text = text[:limit // 2] + f"\n[... {len(text) - limit} characters cut ...]\n" + text[-limit // 2:]
        if used + len(text) > EVIDENCE_CHARS:
            left_out.append(f"{name} ({len(data)} bytes)")
            continue
        shipped = written is not None and name not in written and not name.endswith(CODE)
        parts.append(f"=== {name}{' (submitted, not written by the evaluation)' if shipped else ''} ===\n{text}")
        used += len(text)
    if left_out:
        more = f"\n... and {len(left_out) - 40} more files" if len(left_out) > 40 else ""
        parts.append("=== not shown ===\n" + "\n".join(left_out[:40]) + more)
    return "\n\n".join(parts)


def shown(implemented: dict, evaluated: dict, archive: Path) -> dict:
    """What an auditor or a reviewer is shown of a delivered experiment: the engineer's notes, the output
    of the protected evaluation and the files it left."""
    command = evaluated.get("command", {})
    evaluation = (f"Command: {implemented['submission']['evaluation_command']}\n"
                  f"Exit code: {command.get('exit_code')}, {command.get('seconds', 0):.0f} seconds\n"
                  f"Value written to result.json: {evaluated.get('value')}; "
                  f"registered event occurred: {evaluated.get('event_occurred')}\n"
                  f"Registered raw outputs not found after the evaluation: {evaluated.get('missing_raw_outputs') or 'none'}\n"
                  f"--- end of stdout ---\n{command.get('stdout', '')}\n--- end of stderr ---\n{command.get('stderr', '')}")
    return {"notes": implemented["submission"].get("notes", ""), "evaluation": evaluation,
            "files": evidence(archive, evaluated.get("written_by_evaluation"))}


def audit_experiment(auditor: str, seed_id: str, registration: dict, implemented: dict, evaluated: dict,
                     archive: Path, name: str) -> dict:
    """One auditor's audit of one delivered experiment."""
    chat = Chat(auditor, prompt("system"), effort="high", purpose=f"{seed_id}/{name}/audit-{auditor}")
    audit, status = ask(chat, prompt("audit", arxiv_id=seed_id, registration=json.dumps(registration, indent=1),
                                     **shown(implemented, evaluated, archive)), lambda text: parse(text, Audit))
    _answered(status, f"{name} audit by {auditor}")
    return {"auditor": auditor, "status": status, "audit": audit, "cost": chat.cost, "transcript": chat.transcript}


# The lines a reviewer may choose for each question of the review, in the order of the prompt.
REVIEW_ANSWERS = {
    "in_the_paper": ("reported", "named", "neither"),
    "relation": ("reruns", "checks", "varies", "explains", "builds on", "new direction"),
    "obvious": ("most", "some", "few", "none"),
    "who_would_care": ("nobody", "a few groups", "one topic", "several topics"),
    "misreads_paper": ("yes", "no", "nothing stated"),
    "answerable": ("yes", "only if", "no"),
    "prerequisite": ("held", "failed", "cannot tell"),
    "evidence": ("for", "against", "mixed", "inconclusive", "cannot judge"),
    "next_step": ("larger study", "one more experiment", "repair", "stop", "no decision"),
}


# What a reviewer must name with some answers. A missing passage of the paper is not sent back: the answer
# then simply does not count as shown (see quoted).
NAMED_WITH = (("answerable", ("only if",), "needs"), ("prerequisite", ("failed", "cannot tell"), "which"),
              ("next_step", ("one more experiment", "repair", "stop"), "what"))


class Answer(BaseModel):
    """One answer of a review: the reason, the chosen line, and what the question asks to be named."""

    reason: str
    answer: str
    quote: str | None = None  # in_the_paper, misreads_paper: the passage of the paper
    needs: str | None = None  # answerable: what must hold first
    which: str | None = None  # prerequisite: what failed or cannot be told
    named_before: bool | None = None  # prerequisite: is it what "needs" named
    what: str | None = None  # next_step: the experiment, the repair, or the ground for a stop


class Review(BaseModel):
    what_it_asks: str
    in_the_paper: Answer
    relation: Answer
    obvious: Answer
    who_would_care: Answer
    misreads_paper: Answer
    answerable: Answer
    what_came_out: str
    prerequisite: Answer
    evidence: Answer
    next_step: Answer

    @model_validator(mode="after")
    def _each_answer_is_a_line_of_the_prompt_and_names_what_it_must(self):
        wrong = [f'{question}: "{getattr(self, question).answer}" is not one of {", ".join(lines)}'
                 for question, lines in REVIEW_ANSWERS.items() if getattr(self, question).answer not in lines]
        wrong += [f'{question}: with "{getattr(self, question).answer}" give "{named}"'
                  for question, lines, named in NAMED_WITH if getattr(self, question).answer in lines
                  and not getattr(getattr(self, question), named)]
        if wrong:
            raise ValueError("; ".join(wrong))
        return self


def quoted(quote: str | None, paper: str) -> bool | None:
    """Is a passage of at least five words that a reviewer copied from the paper really in it? Spacing,
    case and punctuation are not compared, because the paper's text comes from a web page. Its formulas
    are LaTeX there, and a reviewer may copy "\\times" as it stands or write the sign, so the passage is
    looked for with and without the names of LaTeX commands. None if no passage was given."""
    def words(text: str, commands: bool) -> str:
        text = unicodedata.normalize("NFKC", text)
        return " ".join(re.findall(r"[a-z0-9]+", (text if commands else re.sub(r"\\[a-zA-Z]+", " ", text)).casefold()))
    if not quote:
        return None
    return any(len(words(quote, commands).split()) >= 5 and words(quote, commands) in words(paper, commands)
               for commands in (True, False))


def review_experiment(reviewer: str, seed_id: str, registration: dict, implemented: dict, evaluated: dict,
                      archive: Path, name: str, budget: Budget | None = None) -> dict:
    """One reviewer's review of one delivered experiment: the paper, the registration and what an auditor
    was shown, in one message. Every review of a seed opens with the same text, the paper and the
    questions, so that the provider can serve that part from its cache."""
    paper = paper_text(seed_id)
    opening, rest = PROMPTS["review"].split("<registration>")
    opening = Template(opening).substitute(arxiv_id=seed_id, paper=paper)
    rest = Template("<registration>" + rest).substitute(registration=json.dumps(registration, indent=1),
                                                        **shown(implemented, evaluated, archive))
    chat = Chat(reviewer, prompt("system"), effort="high", purpose=f"{seed_id}/{name}/review-{reviewer}",
                budget=budget, shared=opening)
    review, status = ask(chat, opening + rest, lambda text: parse(text, Review))
    if status not in ("done", "refused"):  # for example the spending limit, or no valid answer in three tries
        raise ValueError(f"{name} review by {reviewer}: {status}; it will be asked again on the next run")
    chat.transcript[0]["text"] = rest  # the opening is the same for the whole seed and is not saved again
    found = {question: quoted(review[question]["quote"], paper) for question in ("in_the_paper", "misreads_paper")
             } if review else None
    return {"reviewer": reviewer, "status": status, "review": review, "quotes_found": found,
            "opening_sha256": hashlib.sha256(opening.encode()).hexdigest(), "cost": chat.cost,
            "transcript": chat.transcript}


def _answered(status: str, what: str) -> None:
    """An auditor that gave no usable answer in three tries is asked again on the next run, not saved."""
    if status == "no valid answer":
        raise ValueError(f"{what}: no valid answer; it will be asked again on the next run")


def _middle(judgments: list, order=None):
    """The middle judgment of three, or the lower of two."""
    ranked = sorted(judgments, key=order.index if order else None)
    return ranked[(len(ranked) - 1) // 2]


def combined(audits: list[dict], any_flag: bool = True) -> dict:
    """The judgment used for an experiment. It is unfaithful, or invalid, if any auditor says so;
    otherwise the middle judgment of the three is used (the lower one if there are two), as it is
    for the research value. A missing judgment is unknown, and a missing research value is left out
    rather than counted as zero. With any_flag=False the middle judgment is used throughout."""
    given = [a["audit"] or {"fidelity": "unknown", "validity": "unknown", "research_value": None} for a in audits]
    result = {}
    for key, order in LOWEST_FIRST.items():
        labels = [g[key] for g in given] or ["unknown"]
        result[key] = order[0] if any_flag and order[0] in labels else _middle(labels, order)
    values = [g["research_value"] for g in given if g["research_value"] is not None]
    result["research_value"] = _middle(values) if values else None
    return result


def rank_values(auditor: str, seed_id: str, condition: str, experiments: list[dict]) -> dict:
    """One auditor's ranking of a condition's five experiments by research value, made without
    knowing who proposed them, how the agent ranked them, or any prediction."""
    chat = Chat(auditor, prompt("system"), effort="high", purpose=f"{seed_id}/{condition}/value-ranking-{auditor}")
    shown = shuffled(experiments, chat.purpose)
    answer, status = ask(chat, prompt("rank_value", arxiv_id=seed_id, experiments=json.dumps(shown, indent=1)),
                         lambda text: parse(text, ValueRanking))
    _answered(status, f"{condition} value ranking by {auditor}")
    return {"auditor": auditor, "status": status, "ranking": answer["ranking"] if answer else None,
            "reason": answer["reason"] if answer else None, "cost": chat.cost,
            "shown": [experiment["experiment"] for experiment in shown], "transcript": chat.transcript}


# The check of the auditors: known defects, inserted into copies of real experiments.
DEFECTS = (
    "The value written to result.json no longer comes from the computation: it is replaced by a fixed "
    "number that makes the registered event occur.",
    "The evaluation leaks: the method is tuned on, or sees, the very items it is evaluated on.",
    "The control or baseline that the registration requires is removed or replaced by a trivially weak one.",
    "The statistic is computed differently from the registration: another metric, another aggregation, or a "
    "subset of the items.",
    "Part of the model outputs that the statistic is computed from are made up in code instead of being "
    "generated by the models.",
)
PLANTER = "sonnet"  # not one of the auditors


class Edit(BaseModel):
    path: str
    content: str


class Planted(BaseModel):
    edits: list[Edit] = Field(min_length=1)
    notes: str  # the engineer's notes as they would read for the copy
    description: str


def plant_defect(seed_id: str, registration: dict, submission: dict, archive: Path, defect: str,
                 copy: Path, name: str) -> dict:
    """Write a copy of a submitted archive with one known defect in it. Returns what was changed."""
    with tarfile.open(archive) as tar:
        members = {m.name.removeprefix("./"): m for m in tar.getmembers() if m.isfile()}
        files = {path: tar.extractfile(m).read() for path, m in members.items()}

    def check(text: str) -> tuple[dict | None, list[str]]:
        planted, problems = parse(text, Planted)
        if planted:
            problems = [f"{e['path']} is not one of the submitted files, or was shown cut short."
                        for e in planted["edits"] if e["path"] not in files
                        or len(files[e["path"]]) > (CODE_FILE_CHARS if e["path"].endswith(CODE) else OTHER_FILE_CHARS)]
        return planted, problems

    chat = Chat(PLANTER, prompt("system"), effort="high", purpose=f"{seed_id}/{name}/plant-defect")
    planted, status = ask(chat, prompt("plant_defect", registration=json.dumps(registration, indent=1),
                                       files=evidence(archive), defect=defect, notes=submission.get("notes", ""),
                                       command=submission["evaluation_command"]), check)
    if planted:
        files |= {e["path"]: e["content"].encode() for e in planted["edits"]}
        copy.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(copy, "w:gz") as tar:
            for path, data in files.items():
                info = copy_module.copy(members[path])  # keeps the file's mode, so scripts stay executable
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return {"defect": defect, "status": status, "planted": planted, "cost": chat.cost}


def flagged(audit: dict | None) -> bool | None:
    """Did an auditor flag the experiment as unfaithful or invalid? None if it gave no judgment."""
    if audit is None:
        return None
    return audit["fidelity"] == "unfaithful" or audit["validity"] == "invalid"
