"""Build the released dataset from a run store: flat tables (CSV) and the full text of the judgments (JSONL).

    python -m analysis.build_dataset [run store]      the run store defaults to runs/

Reads the saved steps of the seven seed papers and the spend log, and writes data/ at the top of the
repository; data/README.md describes every file and column. No model is called. Transcripts, the seed
papers' text and the executors' code are not copied.
"""

import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from rtaste import audit, run
from rtaste.predict import JUDGES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"
STORE = Path(sys.argv[1]) if len(sys.argv) > 1 else run.RUNS
run.RUNS = STORE  # run.outcome reads from here
SEEDS = ("2609.29233", "2609.11699", "2609.04199", "2609.26796", "2609.18605", "2608.02602", "2608.06867")  # run order
HARNESS = {"single": "SR", "team": "RT"}
AUDITORS = audit.AUDITORS
REVIEWERS = ("grok", "opus", "sol")  # the auditor models; Grok first, as its provider has no model in the study
QUESTIONS = tuple(audit.REVIEW_ANSWERS)
PROVIDER = {"astra": "OpenAI", "sol": "OpenAI", "luna": "OpenAI", "fable": "Anthropic", "opus": "Anthropic",
            "sonnet": "Anthropic", "muse": "Meta", "grok": "xAI", "a100-large": "Hugging Face", "h200": "Hugging Face"}
AT_A100_RATE = {"h200": 2.50 / 5.00}  # seed 7 ran on H200 machines at twice the price of an A100
PROPOSING = ("researcher", "proposer1", "proposer2", "critic", "lead")
STAGES = {"judge": ("judges", "judge"), "follow": ("follow up matching", "novelty judge"),
          "novelty": ("novelty", "novelty judge"), "value": ("value ranking", "auditor"),
          "audit": ("audits", "auditor"), "review": ("reviews", "reviewer"), "executor": ("implementation", "executor"),
          "machine": ("implementation machine", "machine"), "evaluation": ("rerun machine", "machine")}
ATTEMPT_COLUMNS = ("executor", "status", "minutes", "calls", "model_cost", "machine_cost", "output_tokens",
                   "wait_for_machine_minutes")
KEY_LIKE = re.compile(r"\b(?:hf_|sk-ant-|sk-proj-|xai-|ghp_|gho_)[A-Za-z0-9_-]{20,}|\bsk-[A-Za-z0-9]{32,}"
                      r"|\bAKIA[0-9A-Z]{16}\b")


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def when(text: str) -> datetime:
    return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")


def minutes(start: str, end: str) -> float:
    return round((when(end) - when(start)).total_seconds() / 60, 1)


def number(seed: str) -> int:
    return SEEDS.index(seed) + 1


def ids(seed: str, condition: str) -> dict:
    model, policy = condition.split("-")
    return {"seed": seed, "seed_number": number(seed), "condition": condition, "harness": HARNESS[policy],
            "proposer": model}


# ---- calls

def read_calls(checked: str) -> list[dict]:
    """One row per paid model call or machine of the seven seeds, from the spend log."""
    committed = {}  # (seed, condition) -> when its MVEs were committed; the agent's later calls are its forecasts
    for seed in SEEDS:
        for condition in run.CONDITIONS:
            first = load(STORE / seed / condition / "first_commitment.json")
            if first:
                committed[seed, condition] = when(first["committed_at"])
    calls = []
    for line in (STORE / "spend.jsonl").open():
        entry = json.loads(line)
        parts = entry["purpose"].split("/")
        if parts[0] == "auditor-check":  # the check's rerun machines were logged without the seed
            parts = [checked, *parts]
        if parts[0] in SEEDS:
            calls.append(call_row(entry, parts, committed))
    return calls


def call_row(entry: dict, parts: list[str], committed: dict) -> dict:
    seed, last = parts[0], parts[-1]
    row = {"seed": seed, "seed_number": number(seed), "condition": "", "harness": "", "proposer": "", "mve": "",
           "attempt": "", "time": entry["time"], "model": entry["model"], "provider": PROVIDER[entry["model"]],
           "status": entry["status"], "cost": entry["cost"] * AT_A100_RATE.get(entry["model"], 1),
           "cost_paid": entry["cost"],
           "prompt_tokens": sum(entry.get(key) or 0 for key in ("input", "cached", "written")),
           "cached_tokens": entry.get("cached") or 0, "written_tokens": entry.get("written") or 0,
           "output_tokens": entry.get("output") or 0, "machine_minutes": entry.get("minutes", ""), "repeated": False}
    if parts[1] == "auditor-check":
        role = "machine" if last == "evaluation" else "planter" if last == "plant-defect" else "auditor"
        return row | {"stage": "auditor check", "role": role}
    if len(parts) == 2:
        return row | {"stage": "follow up lists", "role": "lister"}
    row |= ids(seed, parts[1])
    if last in PROPOSING or last == "predictions":
        later = when(entry["time"]) > committed.get((seed, parts[1]), datetime.max)
        return row | {"stage": "agent forecasts" if later or last == "predictions" else "proposing", "role": last}
    stage, role = STAGES[last.split("-")[0]]
    if len(parts) >= 4:
        row["mve"] = int(parts[2])
    if len(parts) == 5:
        row["attempt"] = int(parts[3].removeprefix("attempt"))
    return row | {"stage": stage, "role": role}


def by_attempt(calls: list[dict]) -> dict:
    """The executor calls and machines of each attempt. Marks the work that was thrown away as repeated: executor
    calls beyond those the saved attempt holds, a machine started before them, and a rerun machine that died."""
    groups = defaultdict(list)
    for call in calls:
        if call["attempt"]:
            groups[call["seed"], call["condition"], call["mve"], call["attempt"]].append(call)
    for (seed, condition, n, k), entries in groups.items():
        saved = load(STORE / seed / condition / f"mve{n}" / f"attempt{k}.json")
        turns = sum(said["role"] == "assistant" for said in saved["transcript"]) if saved else 0
        talk = [e for e in entries if e["role"] == "executor"]
        for e in talk[:len(talk) - turns]:
            e["repeated"] = True
        kept = [e["time"] for e in talk if not e["repeated"]]
        for e in entries:
            if e["stage"] == "implementation machine" and (not kept or e["time"] < kept[0]):
                e["repeated"] = True
        reruns = [e for e in entries if e["stage"] == "rerun machine"]
        starts = [i for i, e in enumerate(reruns) if e["status"] == "machine started"]
        for e in reruns[:starts[-1]] if starts else []:
            e["repeated"] = True
    return groups


# ---- the form of the registered event: a keyword rule, then corrected by one reader (Claude) where READ says so

BOUND_WORDS = ("lower bound", "upper bound", "confidence interval", "bootstrap", "one-sided")
SEVERAL_WORDS = ("min(", "max(", "min[", "max[", "min of ", "max of ", "minimum of", "minimum over", "maximum over")
GUARD_IN_STATISTIC = ("set to 0", "otherwise", " if ")
GUARD_IN_DESCRIPTION = ("otherwise", "only if", "requires ", "set to 0", "not evaluable", "does not occur", "provided ")
READ = {  # (seed number, condition, MVE): (its properties after reading, why the keywords mislead)
    (2, "fable-single", 1): ("", "max(0, x) only cuts a per-token gate at zero"),
    (2, "fable-team", 1): ("", "max(0, x) only cuts a per-token gate at zero"),
    (2, "astra-single", 5): ("bound", "min(x, 0) is a penalty inside one statistic"),
    (3, "astra-single", 1): ("bound", "min(...) defines the measure: accuracy in the worse of two strata"),
    (3, "astra-single", 4): ("bound", "'minimum of' defines the measure: accuracy in the worst of three regions"),
    (5, "sol-single", 2): ("bound", "max(|L|, |U|) is the two-sided bound itself"),
    (7, "astra-team", 1): ("bound", "max(L, -U) is the two-sided bound itself"),
    (2, "astra-single", 3): ("bound", "'if' and 'otherwise' define the two-sided bound"),
    (2, "astra-team", 3): ("bound", "'if' defines the two-sided bound"),
    (6, "sol-single", 2): ("bound", "'if' defines the two-sided bound"),
    (1, "sol-team", 5): ("bound, several, prerequisite", "the minimum of four margins, over both seeds"),
    (5, "astra-team", 2): ("bound, several", "the largest endpoint across two intervals: both effects must be small"),
    (1, "luna-team", 5): ("bound, prerequisite", "the description asks for 256 kept labels and a passed check")}


def event_form(event: dict, key: tuple) -> str:
    statistic, description = f" {event['statistic'].lower()} ", f" {event['description'].lower()} "
    found = {"bound": any(w in statistic for w in BOUND_WORDS), "several": any(w in statistic for w in SEVERAL_WORDS),
             "prerequisite": any(w in statistic for w in GUARD_IN_STATISTIC)
             or any(w in description for w in GUARD_IN_DESCRIPTION)}
    has = set(filter(None, READ[key][0].split(", "))) if key in READ else {name for name, yes in found.items() if yes}
    return "confidence bound" if "bound" in has else "composite" if has else "plain threshold"


# ---- MVEs

def delivery(seed: str, condition: str, n: int, result: dict | None, attempts: list, groups: dict) -> dict:
    """Who delivered the MVE, on what, and what each attempt took."""
    delivered = bool(result and result["status"] == "delivered")
    evaluated = result["evaluated"] if delivered else {}
    first = attempts[0] or {}
    row = {"attempts": sum(a is not None for a in attempts), "delivered": delivered,
           "delivered_on_attempt": result["attempt"] if delivered else "",
           "executor": result["implemented"]["executor"] if delivered else first.get("executor", ""),
           "assigned_executor": first.get("executor", ""), "implemented_on": "", "rerun_on": "",
           "first_attempt_status": first.get("status", ""), "value": evaluated.get("value", ""),
           "executor_value": evaluated.get("executor_value", ""), "event": evaluated.get("event_occurred", ""),
           "rerun_seconds": round(evaluated["command"]["seconds"]) if delivered else "",
           "missing_raw_outputs": len(evaluated.get("missing_raw_outputs") or []) if delivered else ""}
    for k, attempt in enumerate(attempts, 1):
        if not attempt:
            row |= {f"attempt{k}_{column}": "" for column in ATTEMPT_COLUMNS}
            continue
        entries = [e for e in groups.get((seed, condition, n, k), []) if not e["repeated"]]
        talk = [e for e in entries if e["role"] == "executor"]
        machine = [e for e in entries if e["stage"] == "implementation machine" and e["status"] == "machine started"]
        rerun = [e for e in entries if e["stage"] == "rerun machine" and e["status"] == "machine started"]
        asked = [e["time"] for e in talk if machine and e["time"] <= machine[0]["time"]]
        flavor = machine[0]["model"] if machine else ""
        if delivered and result["attempt"] == k:
            row |= {"implemented_on": flavor, "rerun_on": rerun[0]["model"] if rerun else ""}
        values = (attempt["executor"], attempt["status"], round(attempt["minutes"], 1), len(talk),
                  round(attempt["model_cost"], 3), round(attempt["compute_cost"] * AT_A100_RATE.get(flavor, 1), 3),
                  sum(e["output_tokens"] for e in talk), minutes(asked[-1], machine[0]["time"]) if asked else "")
        row |= {f"attempt{k}_{column}": value for column, value in zip(ATTEMPT_COLUMNS, values)}
    return row


def audit_columns(audits: dict, rankings: dict, n: int) -> dict:
    """Each auditor's judgment and value rank, and the rule over the three (sound: at least two say faithful and
    at least two say valid; the research value is the median)."""
    row = {}
    for a in AUDITORS:
        said = audits.get(a) or {}
        row |= {f"fidelity_{a}": said.get("fidelity", ""), f"validity_{a}": said.get("validity", ""),
                f"research_value_{a}": said.get("research_value", ""),
                f"value_rank_{a}": rankings[a].index(n) + 1 if rankings[a] else ""}
    middle = audit.combined([{"audit": audits.get(a)} for a in AUDITORS], any_flag=False) if audits else None
    ranks = [rankings[a].index(n) + 1 for a in AUDITORS if rankings[a]]
    return row | {"sound": bool(middle and middle["fidelity"] == "faithful" and middle["validity"] == "valid"),
                  "research_value": middle["research_value"] if middle else 0,
                  "mean_value_rank": round(sum(ranks) / len(ranks), 2) if len(ranks) >= 2 else ""}


def review_columns(reviews: dict) -> dict:
    """Each reviewer's answers, and the labels that at least two of the three reviewers give."""
    row, flags = {}, {}
    for r in REVIEWERS:
        saved = reviews.get(r) or {}
        review, found = saved.get("review") or {}, saved.get("quotes_found") or {}
        answer = {q: (review.get(q) or {}).get("answer", "") for q in QUESTIONS}
        row |= {f"{q}_{r}": answer[q] for q in QUESTIONS}
        row[f"in_the_paper_quote_found_{r}"] = found.get("in_the_paper")
        in_paper = answer["in_the_paper"] in ("reported", "named") and bool(found.get("in_the_paper"))
        beyond = answer["relation"] in ("builds on", "new direction")
        non_obvious = answer["obvious"] in ("few", "none") and not in_paper
        flags[r] = {"beyond": beyond, "non_obvious": non_obvious, "bno": beyond and non_obvious,
                    "informative": answer["evidence"] in ("for", "against", "mixed")}

    def majority(question: str) -> str:
        answers = [row[f"{question}_{r}"] for r in REVIEWERS]
        return next((a for a in answers if answers.count(a) >= 2), "")

    def two(flag: str) -> bool:
        return sum(flags[r][flag] for r in REVIEWERS) >= 2

    return row | {"relation_2of3": majority("relation"), "check_2of3": majority("relation") == "checks",
                  "beyond_2of3": two("beyond"), "non_obvious_2of3": two("non_obvious"), "bno_2of3": two("bno"),
                  "informative_2of3": two("informative"), "evidence_2of3": majority("evidence"),
                  "next_step_2of3": majority("next_step")}


def forecasters(second: dict) -> dict:
    content = second["content"]
    return {"agent": content["agent"], **{f"judge_{j}": content["judges"].get(j) for j in JUDGES}}


def forecast(said: dict | None, n: int) -> dict | None:
    if not said:
        return None
    given = next(p for p in said["predictions"] if p["experiment"] == n)
    return {"probability": given["probability"], "rank": said["ranking"].index(n) + 1, "reason": given["reason"]}


def condition_mves(seed: str, condition: str, groups: dict) -> tuple[list[dict], list[dict]]:
    """The rows of mves.csv and the records of mves.jsonl for one condition's five MVEs."""
    folder = STORE / seed / condition
    first, second = run.commitments(folder)
    rankings = {a: (load(folder / f"value_ranking_{a}.json") or {}).get("ranking") for a in AUDITORS}
    matching = {a: load(folder / f"follow_ups_{a}.json") for a in audit.NOVELTY_JUDGES}
    rows, records = [], []
    for n, registration in enumerate(first["content"]["registrations"], 1):
        result = run.outcome(seed, condition, n)
        delivered = bool(result and result["status"] == "delivered")
        attempts = [load(folder / f"mve{n}" / f"attempt{k}.json") for k in (1, 2)]
        novelty = {a: load(folder / f"novelty_{n}_{a}.json") for a in audit.NOVELTY_JUDGES}
        matched = {a: next((m for m in matching[a]["matches"] or [] if m["experiment"] == n), None)
                   for a in audit.NOVELTY_JUDGES}
        audits = {a: load(folder / f"mve{n}" / f"audit_{a}.json") for a in AUDITORS} if delivered else {}
        reviews = {r: load(STORE / seed / "review" / f"{condition}-mve{n}" / f"review_{r}.json")
                   for r in REVIEWERS} if delivered else {}
        event = registration["event"]
        row = ids(seed, condition) | {
            "mve": n, "title": registration["title"], "relation": event["relation"], "threshold": event["threshold"],
            "event_form": event_form(event, (number(seed), condition, n)),
            "expected_minutes": registration["resources"]["expected_minutes"],
            "needs_gpu": registration["resources"]["needs_gpu"],
            "models_registered": "; ".join(registration["resources"]["models"])}
        row |= delivery(seed, condition, n, result, attempts, groups)
        for a in audit.NOVELTY_JUDGES:
            row |= {f"novelty_{a}": ((novelty[a] or {}).get("novelty") or {}).get("label", ""),
                    f"on_list_{a}": matched[a]["same_as"] is not None if matched[a] else ""}
        row |= audit_columns({a: saved["audit"] for a, saved in audits.items()}, rankings, n)
        row["valuable"] = (row["research_value"] or 0) >= 2
        row["new"] = audit.novelty_label(list(novelty.values())) == "distinct"
        row["good"] = delivered and row["sound"] and row["valuable"] and row["new"]
        row["on_list"] = True in (row["on_list_opus"], row["on_list_sol"])
        for who, said in forecasters(second).items():
            given = forecast(said, n) or {}
            row |= {f"{who}_probability": given.get("probability", ""), f"{who}_rank": given.get("rank", "")}
        rows.append(row | review_columns(reviews))
        records.append(ids(seed, condition) | {
            "mve": n, "registration": registration,
            "forecasts": {who: forecast(said, n) for who, said in forecasters(second).items()},
            "novelty": {a: {"status": saved["status"], **(saved["novelty"] or {})} for a, saved in novelty.items()},
            "follow_up_match": {a: {"status": matching[a]["status"], "same_as": (matched[a] or {}).get("same_as"),
                                    "reason": (matched[a] or {}).get("reason")} for a in audit.NOVELTY_JUDGES},
            "outcome": outcome_record(folder / f"mve{n}", result, attempts),
            "audits": {a: {"status": saved["status"], **(saved["audit"] or {})} for a, saved in audits.items()},
            "reviews": {r: {"status": saved["status"], **(saved["review"] or {}), "quotes_found": saved["quotes_found"]}
                        for r, saved in reviews.items()}})
    shares = most_valued_shares(rows)
    for row in rows:
        row["auditors_first"] = shares.get(row["mve"], "")
    return rows, records


def outcome_record(folder: Path, result: dict | None, attempts: list) -> dict:
    """What each attempt ended in and, for a delivered MVE, what its rerun found and the executor's notes."""
    tried = [{"attempt": k, "executor": attempt["executor"], "status": attempt["status"], "reason": attempt["reason"],
              "rerun": (load(folder / f"attempt{k}_evaluation.json") or {}).get("status"),
              "rerun_reason": (load(folder / f"attempt{k}_evaluation.json") or {}).get("reason")}
             for k, attempt in enumerate(attempts, 1) if attempt]
    if not result or result["status"] != "delivered":
        return {"delivered": False, "attempts": tried}
    evaluated = result["evaluated"]
    return {"delivered": True, "attempts": tried, "attempt": result["attempt"],
            "executor": result["implemented"]["executor"], "value": evaluated["value"],
            "executor_value": evaluated.get("executor_value"), "event_occurred": evaluated["event_occurred"],
            "rerun_seconds": round(evaluated["command"]["seconds"]),
            "missing_raw_outputs": evaluated.get("missing_raw_outputs") or [],
            "executor_notes": result["implemented"]["submission"].get("notes", "")}


def most_valued_shares(rows: list[dict]) -> dict:
    """Each MVE's share of the auditors' first place: the best mean value rank, shared equally by a tie."""
    ranks = {row["mve"]: row["mean_value_rank"] for row in rows if row["mean_value_rank"] != ""}
    top = [n for n, rank in ranks.items() if rank == min(ranks.values())]
    return {n: 1 / len(top) if n in top else 0.0 for n in ranks}


# ---- conditions, seeds, rankings, lists, check of the auditors

def condition_row(seed: str, condition: str, calls: list[dict], rows: list[dict]) -> dict:
    folder = STORE / seed / condition
    proposal = load(folder / "proposal.json")
    mine = [c for c in calls if c["seed"] == seed and c["condition"] == condition]
    proposing = [c for c in mine if c["stage"] == "proposing"]
    forecasting = [c for c in mine if c["stage"] == "agent forecasts"]
    tools = Counter(call["name"] for transcript in (proposal.get("transcripts") or {}).values()
                    for said in transcript for call in said.get("calls") or [])
    first = load(folder / "first_commitment.json")
    row = ids(seed, condition) | {
        "status": "committed" if proposal["status"] == "done" else proposal["status"],
        "proposing_cost": sum(c["cost"] for c in proposing), "proposing_calls": len(proposing),
        "proposing_prompt_tokens": sum(c["prompt_tokens"] for c in proposing),
        "proposing_output_tokens": sum(c["output_tokens"] for c in proposing),
        "arxiv_searches": tools["arxiv_search"], "papers_read": tools["arxiv_read"],
        "repo_listings": tools["repo_ls"], "repo_reads": tools["repo_read"],
        "replies_cut_off": sum(c["status"] == "truncated" for c in proposing),
        "minutes_first_reply_to_commitment": minutes(min(c["time"] for c in proposing), first["committed_at"])
        if first and proposing else "",
        "forecasting_minutes": minutes(first["committed_at"], max(c["time"] for c in forecasting))
        if first and forecasting else "",
        "condition_cost": condition_cost(folder, mine)}
    counts = ("delivered", "sound", "new", "on_list", "good")
    if not rows:  # refused before committing: nothing registered, nothing delivered
        return row | {name: 0 for name in counts} | {"events": 0}
    sound = [m for m in rows if m["sound"]]
    row |= {name: sum(m[name] for m in rows) for name in counts}
    row |= {"on_list_sol_alone": sum(m["on_list_sol"] is True for m in rows) if rows[0]["on_list_sol"] != "" else "",
            "mean_value": mean([m["research_value"] for m in rows]),
            "value_rankers": sum(rows[0][f"value_rank_{a}"] != "" for a in AUDITORS),
            "events": sum(m["event"] is True for m in sound), "scored": len(sound)}
    model = condition.split("-")[0]
    row |= {"brier_agent": brier(sound, "agent"), "brier_same_model_judge": brier(sound, f"judge_{model}")}
    row |= {f"brier_judge_{j}": brier(sound, f"judge_{j}") for j in JUDGES}

    def first_is_most_valued(rank_column: str):
        return next((m["auditors_first"] for m in rows if m[rank_column] == 1), "")

    row |= {"agent_first_is_most_valued": first_is_most_valued("agent_rank"),
            "first_written_is_most_valued": rows[0]["auditors_first"],
            "same_model_judge_first_is_most_valued": first_is_most_valued(f"judge_{model}_rank")}
    return row | {f"judge_{j}_first_is_most_valued": first_is_most_valued(f"judge_{j}_rank") for j in JUDGES}


def mean(values: list):
    values = [v for v in values if v is not None and v != ""]
    return sum(values) / len(values) if values else ""


def brier(sound: list[dict], who: str):
    """Mean squared error of a forecaster's probabilities over the sound MVEs, whose registered event is scored."""
    scored = [m for m in sound if m[f"{who}_probability"] != ""]
    return sum((m[f"{who}_probability"] - m["event"]) ** 2 for m in scored) / len(scored) if scored else ""


def condition_cost(folder: Path, mine: list[dict]) -> float:
    """Proposing, the agent's forecasts, and every attempt and rerun of its MVEs, with H200 time at the A100 rate."""
    total = sum((load(folder / name) or {}).get("cost", 0.0) for name in ("proposal.json", "agent_predictions.json"))
    for path in folder.glob("mve*/attempt?.json"):
        record = load(path)
        total += record["model_cost"] + record["compute_cost"]
    total += sum(load(path)["compute_cost"] for path in folder.glob("mve*/attempt?_evaluation.json"))
    return total - sum(c["cost_paid"] - c["cost"] for c in mine if not c["repeated"])


def seed_row(seed: str, calls: list[dict], posted: dict) -> dict:
    info = load(STORE / seed / "seed.json")
    mine = [c for c in calls if c["seed"] == seed and c["stage"] != "auditor check"]
    return {"seed": seed, "seed_number": number(seed), "title": info["title"], "pages": info["pages"],
            "first_posted": posted[seed], "launch_day": info["launch_day"], "repository": info["code"],
            "repository_commit": info["code_commit"],
            "machines": "/".join(sorted({c["model"] for c in mine if c["provider"] == "Hugging Face"})),
            "cost": round(sum(c["cost"] for c in mine), 2), "cost_paid": round(sum(c["cost_paid"] for c in mine), 2)}


def ranking_record(seed: str, condition: str) -> dict:
    """The agent's and each judge's ranking of a condition's five MVEs, and each auditor's value ranking."""
    folder = STORE / seed / condition
    rankers = forecasters(run.commitments(folder)[1])
    record = ids(seed, condition) | {who: {"ranking": said["ranking"], "ranking_reason": said["ranking_reason"]}
                                     if said else None for who, said in rankers.items()}
    for a in AUDITORS:
        saved = load(folder / f"value_ranking_{a}.json")
        record[f"auditor_{a}"] = {"status": saved["status"], "ranking": saved["ranking"],
                                  "ranking_reason": saved["reason"]}
    return record


def follow_up_lists(seed: str) -> list[dict]:
    """Each model's ten follow ups, numbered as in the pooled list the novelty judges matched MVEs against."""
    records, start = [], 0
    for model in JUDGES:  # pooled in this order
        listed = load(STORE / seed / f"follow_ups_{model}.json")
        follow_ups = [{"number": start + i, **e} for i, e in enumerate(listed["experiments"], 1)]
        start += len(follow_ups)
        records.append({"seed": seed, "seed_number": number(seed), "model": model, "status": listed["status"],
                        "follow_ups": follow_ups})
    return records


def auditor_check(seed: str) -> list[dict]:
    """One row per copy of a delivered MVE with a planted defect, with what each auditor said of it and whether it
    flagged the original (flagged: judged unfaithful or invalid)."""
    rows = []
    for folder in sorted((STORE / seed / "auditor_check").iterdir()):
        condition, n = folder.name.rsplit("-mve", 1)
        planted, evaluated = load(folder / "planted.json"), load(folder / "evaluation.json") or {}
        row = {"seed": seed, "seed_number": number(seed), "condition": condition, "mve": int(n),
               "defect": audit.DEFECTS.index(planted["defect"]) + 1,
               "change": (planted["planted"] or {}).get("description", ""), "rerun": evaluated.get("status", ""),
               "value": evaluated.get("value", ""), "event": evaluated.get("event_occurred", "")}
        for a in AUDITORS:
            copy = (load(folder / f"audit_{a}.json") or {}).get("audit")
            original = (load(STORE / seed / condition / f"mve{n}" / f"audit_{a}.json") or {}).get("audit")
            row |= {f"{key}_{a}": (copy or {}).get(key, "") for key in ("fidelity", "validity", "research_value")}
            row |= {f"flagged_{a}": audit.flagged(copy), f"original_flagged_{a}": audit.flagged(original)}
        row["flagged_2of3"] = sum(row[f"flagged_{a}"] is True for a in AUDITORS) >= 2
        row["original_flagged_2of3"] = sum(row[f"original_flagged_{a}"] is True for a in AUDITORS) >= 2
        rows.append(row)
    return rows


# ---- writing

def tidy(value):
    return round(value, 12) if isinstance(value, float) else value  # drops float noise, keeps every real digit


def write(name: str, text: str, rows: int) -> None:
    if KEY_LIKE.search(text):
        sys.exit(f"{name}: something that looks like a key would be written. Remove it from the run store first.")
    (OUT / name).write_text(text)
    print(f"{name}: {rows} rows, {len(text.encode()) / 1e6:.1f} MB")


def write_csv(name: str, rows: list[dict]) -> None:
    text = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: tidy(value) for key, value in row.items()} for row in rows)
    write(name, text.getvalue(), len(rows))


def write_jsonl(name: str, records: list[dict]) -> None:
    write(name, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), len(records))


def main() -> None:
    checked = next(seed for seed in SEEDS if (STORE / seed / "auditor_check").exists())  # the check ran on one seed
    calls = read_calls(checked)
    groups = by_attempt(calls)
    shortlist = json.loads((ROOT / "seeds" / "shortlist.json").read_text())
    posted = {paper["id"]: paper["first_posted"] for paper in shortlist["run_order"]}
    mves, records, conditions, rankings, lists = [], [], [], [], []
    for seed in SEEDS:
        for condition in run.CONDITIONS:
            rows = []
            if (STORE / seed / condition / "second_commitment.json").exists():
                rows, saved = condition_mves(seed, condition, groups)
                mves += rows
                records += saved
                rankings.append(ranking_record(seed, condition))
            conditions.append(condition_row(seed, condition, calls, rows))
        lists += follow_up_lists(seed)
    OUT.mkdir(exist_ok=True)
    write_csv("seeds.csv", [seed_row(seed, calls, posted) for seed in SEEDS])
    write_csv("conditions.csv", conditions)
    write_csv("mves.csv", mves)
    write_csv("calls.csv", calls)
    write_csv("auditor_check.csv", auditor_check(checked))
    write_jsonl("mves.jsonl", records)
    write_jsonl("rankings.jsonl", rankings)
    write_jsonl("follow_up_lists.jsonl", lists)


if __name__ == "__main__":
    main()
