"""Run the study for one seed paper, one step at a time.

    python -m rtaste.run propose 2608.31119      everything before execution, for all twelve conditions
    python -m rtaste.run execute 2608.31119      implement and evaluate every committed MVE
    python -m rtaste.run execute 2608.31119 2    only the first two MVEs of each condition (the rehearsal)
    python -m rtaste.run audit 2608.31119        audit every delivered MVE and rank each condition's five
    python -m rtaste.run check-auditors 2609.29233 10   plant known defects in copies of 10 audited MVEs
    python -m rtaste.run review 2608.31119       the review added after the study, of every delivered MVE
    python -m rtaste.run status 2608.31119       what is done and what it has cost

Each finished step is saved as one JSON file under runs/<seed>/ and is never computed again, so a
run that stops can simply be started again. Nothing under runs/ is overwritten.

A seed is run from committed code, and seed.json names the commit. If the code changes while a seed
is under way, the run stops; add --new-code to go on with the new code and have the change recorded.
"""

import fcntl
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rtaste import audit, execute, predict
from rtaste.models import ROOT, SPEND_LOG, Budget, Chat, now
from rtaste.propose import PROPOSER_CAP, Proposal, propose_single, propose_team
from rtaste.tools import seed_inputs

RUNS = ROOT / "runs"
PARALLEL_MVES = 16  # MVEs executed at the same time, each on its own machine
PROPOSERS = predict.JUDGES  # the models that propose are also the fresh judges
CONDITIONS = [f"{model}-{policy}" for model in PROPOSERS for policy in ("single", "team")]
REVIEW_CAP = 110.0  # dollars the reviews of one seed may cost in all, over every run of the stage
FAILED: list[str] = []  # tasks that raised in this run; the command then ends with an error


def step(path: Path, make):
    """Return the saved result of a step, or carry the step out and save it."""
    if path.exists():
        return json.loads(path.read_text())
    result = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    unfinished = path.with_suffix(".unfinished")
    unfinished.write_text(json.dumps(result, indent=1))
    unfinished.rename(path)
    return result


def in_parallel(function, items, workers: int = 10) -> list:
    """Run one failure-tolerant task per item. A task that raises is reported and gives None."""
    def safely(item):
        try:
            return function(item)
        except Exception as error:
            print(f"  FAILED {str(item)[:80]}: {type(error).__name__}: {error}", flush=True)
            FAILED.append(str(item)[:80])
            return None

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(safely, items))


def code_version() -> str:
    """The last commit that changed the code, the prompts or the seed list. A seed is run only from
    committed code, so its results name what made them."""
    def git(*words: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *words], check=True, capture_output=True, text=True).stdout.strip()

    changed = git("status", "--porcelain", "--", "rtaste", "prompts", "seeds")
    if changed:
        sys.exit(f"Commit the changes to rtaste/, prompts/ and seeds/ before running a seed:\n{changed}")
    return git("log", "-1", "--format=%H", "--", "rtaste", "prompts", "seeds")


def check_unchanged(seed: dict) -> None:
    """Stop if the seed paper or the code is no longer what this seed was started with, so that one
    seed is never a silent mix of versions. With --new-code the run goes on with the new code and the
    change is written to the seed's code_changes.jsonl."""
    if "code_version" not in seed:  # a seed from before versions were recorded
        return
    inputs = seed_inputs(seed["seed"])
    if any(seed[name] != value for name, value in inputs.items()):
        sys.exit(f"The paper is not the one seed {seed['seed']} was started with: now {inputs}")
    log = RUNS / seed["seed"] / "code_changes.jsonl"
    changes = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    was, version = (changes[-1] if changes else seed)["code_version"], code_version()
    if version == was:
        return
    if "--new-code" not in sys.argv:
        sys.exit(f"Seed {seed['seed']} was run with code {was[:7]} and the code is now {version[:7]}. Check out "
                 f"{was[:7]}, or add --new-code to go on with the new code and record the change.")
    with log.open("a") as file:
        file.write(json.dumps({"time": now(), "code_version": version}) + "\n")


def commitments(folder: Path) -> tuple[dict, dict]:
    """A condition's two commitments, checked: neither was changed after it was made, and the second
    names the first. Execution, the audits and the analysis take the experiments and predictions from here."""
    first, second = (json.loads((folder / f"{which}_commitment.json").read_text()) for which in ("first", "second"))
    if not (predict.intact(first) and predict.intact(second)
            and second["content"]["first_commitment"] == first["sha256"]):
        raise ValueError(f"{folder}: a commitment does not match its hash")
    return first, second


def start_seed(seed_id: str) -> dict:
    """Fix the launch day, the code version and the seed's paper and code. All are decided once, when
    the seed is first run."""
    folder = RUNS / seed_id
    shortlist = json.loads((ROOT / "seeds" / "shortlist.json").read_text())
    paper = next(p for p in shortlist["run_order"] + shortlist["spares_in_order"] if p["id"] == seed_id)

    def make() -> dict:
        repo = folder / "repo"
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", paper["code"], str(repo)], check=True)
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        # The executors alternate across seeds in the order the seeds are started (Section 4).
        position = len(list(RUNS.glob("*/seed.json")))
        return {"seed": seed_id, "title": paper["title"], "code": paper["code"], "code_commit": commit,
                "launch_day": now()[:10], "position": position, "code_version": code_version(),
                **seed_inputs(seed_id)}

    return step(folder / "seed.json", make)


def propose_and_predict(seed: dict, condition: str) -> dict:
    """One condition, from reading the seed to the first commitment and the agent's own predictions.

    The agent predicts in the conversation that proposed, so these steps run in one go. If the
    run stops between them, the conversation is restored from the saved proposal.
    """
    folder = RUNS / seed["seed"] / condition
    model, policy = condition.split("-")
    live: list[Proposal] = []

    def make_proposal() -> dict:
        propose = propose_single if policy == "single" else propose_team
        proposal = propose(model, seed["seed"], RUNS / seed["seed"] / "repo", seed["launch_day"])
        live.append(proposal)
        return {"condition": condition, "status": proposal.status, "registrations": proposal.registrations,
                "before_search": proposal.before_search, "drafts": proposal.drafts, "critique": proposal.critique,
                "cost": proposal.cost, "transcripts": proposal.transcripts,
                "chat": {"system": proposal.chat.system, "tools": proposal.chat.tools,
                         "history": proposal.chat.history} if proposal.chat else None}

    saved = step(folder / "proposal.json", make_proposal)
    if saved["status"] != "done":
        return saved
    if live:
        proposal = live[0]
    else:  # restore the conversation, with what the proposal left of its spending limit
        chat = Chat(model, saved["chat"]["system"], saved["chat"]["tools"], effort="max",
                    purpose=f"{seed['seed']}/{condition}/predictions", budget=Budget(PROPOSER_CAP, saved["cost"]))
        chat.history = saved["chat"]["history"]
        proposal = Proposal(condition, "done", saved["registrations"], chat=chat)
    first = step(folder / "first_commitment.json", lambda: predict.first_commitment(seed["seed"], proposal))
    step(folder / "agent_predictions.json", lambda: predict.predict_agent(proposal))
    return {**saved, "first_commitment": first}


def measure(seed: dict, condition: str, follow_ups: list[dict]) -> None:
    """What others say about a condition's five committed experiments before any of them is run."""
    seed_id, launch_day = seed["seed"], seed["launch_day"]
    folder = RUNS / seed_id / condition
    first = json.loads((folder / "first_commitment.json").read_text())
    if not predict.intact(first):
        raise ValueError(f"{folder}: the first commitment does not match its hash")
    registrations = first["content"]["registrations"]
    jobs = [(folder / f"judge_{judge}.json",
             lambda judge=judge: predict.predict_fresh(judge, seed_id, registrations, condition))
            for judge in predict.JUDGES]
    jobs += [(folder / f"novelty_{n}_{auditor}.json",
              lambda n=n, r=r, auditor=auditor: audit.judge_novelty(auditor, seed_id, r, launch_day, f"{condition}/{n}"))
             for n, r in enumerate(registrations, 1) for auditor in audit.NOVELTY_JUDGES]
    jobs += [(folder / f"follow_ups_{auditor}.json",
              lambda auditor=auditor: audit.match_follow_ups(auditor, seed_id, registrations, follow_ups, condition))
             for auditor in audit.NOVELTY_JUDGES]
    in_parallel(lambda job: step(*job), jobs, workers=8)

    # The second commitment closes the condition: it is made once every step above is saved, whether
    # with an answer or with a recorded refusal. A condition without it is not ready for execution.
    if all(path.exists() for path, _ in jobs):
        agent = json.loads((folder / "agent_predictions.json").read_text())
        judges = [json.loads((folder / f"judge_{judge}.json").read_text()) for judge in predict.JUDGES]
        step(folder / "second_commitment.json", lambda: predict.second_commitment(first, agent, judges))


def hold(seed_id: str):
    """Make sure no other run of this seed is going (two at once would pay twice), and that the seed
    is continued with the paper and the code it was started with."""
    folder = RUNS / seed_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"Another run of seed {seed_id} is still going.")
    if (folder / "seed.json").exists():
        check_unchanged(json.loads((folder / "seed.json").read_text()))
    return lock  # the lock lasts as long as this file stays open


def propose(seed_id: str, conditions: list[str]) -> None:
    """Everything that happens for a seed before execution."""
    lock = hold(seed_id)  # noqa: F841
    folder = RUNS / seed_id
    seed = start_seed(seed_id)
    print(f"{seed['title']} (launch day {seed['launch_day']})", flush=True)

    lists = in_parallel(lambda model: step(folder / f"follow_ups_{model}.json",
                                           lambda: audit.list_follow_ups(model, seed_id)), PROPOSERS)
    if None in lists:
        sys.exit("Not every model listed its follow ups. Start the run again.")
    follow_ups = step(folder / "follow_ups.json", lambda: audit.pool_follow_ups(lists))

    proposals = in_parallel(lambda condition: propose_and_predict(seed, condition), conditions)
    committed = [c for c, p in zip(conditions, proposals) if p and p["status"] == "done"]
    in_parallel(lambda condition: measure(seed, condition, follow_ups), committed)
    status(seed_id)


def run_mve(seed: dict, condition: str, number: int, registration: dict, first_executor: str) -> dict | None:
    """Implement and evaluate one MVE: a first attempt and, if it delivers nothing, a second one by the
    other executor. Returns the protected evaluation that delivered, or None."""
    folder = RUNS / seed["seed"] / condition / f"mve{number}"
    second_executor = next(e for e in execute.EXECUTORS if e != first_executor)
    for attempt, executor in enumerate((first_executor, second_executor), 1):
        name = f"{seed['seed']}/{condition}/{number}/attempt{attempt}"
        files = folder / f"attempt{attempt}"
        implemented = step(folder / f"attempt{attempt}.json", lambda: execute.implement(
            executor, seed, registration, RUNS / seed["seed"] / "repo", files, name))
        if implemented["status"] != "submitted":
            continue
        evaluated = step(folder / f"attempt{attempt}_evaluation.json", lambda: execute.evaluate(
            seed, registration, implemented["submission"], files, name))
        if evaluated["status"] == "delivered":
            return evaluated
    return None


def execute_seed(seed_id: str, mves: int = 5) -> None:
    """Execute every committed MVE of a seed, once every condition is ready."""
    lock = hold(seed_id)  # noqa: F841
    seed = json.loads((RUNS / seed_id / "seed.json").read_text())
    execute.stop_leftover_machines(seed_id)
    jobs, waiting = [], []
    for condition in CONDITIONS:
        folder = RUNS / seed_id / condition
        if not (folder / "proposal.json").exists():
            waiting.append(condition)
        elif (folder / "second_commitment.json").exists():
            registrations = commitments(folder)[0]["content"]["registrations"]
            executor = execute.executor_for(condition.split("-")[0], seed["position"])
            jobs += [(condition, n, r, executor) for n, r in enumerate(registrations[:mves], 1)]
        elif json.loads((folder / "proposal.json").read_text())["status"] == "done":
            waiting.append(condition)
    if waiting:  # no outcome may be seen before every prediction for the seed is committed
        sys.exit(f"Not ready for execution: {', '.join(waiting)}. Run propose first.")
    print(f"executing {len(jobs)} MVEs, {PARALLEL_MVES} at a time", flush=True)
    in_parallel(lambda job: run_mve(seed, *job), jobs, workers=PARALLEL_MVES)
    status(seed_id)


def outcome(seed_id: str, condition: str, number: int) -> dict | None:
    """Where an MVE stands after execution: delivered, no delivery, or None while it is still to be run."""
    folder = RUNS / seed_id / condition / f"mve{number}"
    for attempt in (1, 2):
        if not (folder / f"attempt{attempt}.json").exists():
            return None
        implemented = json.loads((folder / f"attempt{attempt}.json").read_text())
        if implemented["status"] != "submitted":
            continue
        if not (folder / f"attempt{attempt}_evaluation.json").exists():
            return None
        evaluated = json.loads((folder / f"attempt{attempt}_evaluation.json").read_text())
        if evaluated["status"] == "delivered":
            return {"status": "delivered", "attempt": attempt, "implemented": implemented, "evaluated": evaluated}
    return {"status": "no delivery"}


def audit_seed(seed_id: str) -> None:
    """Audit every delivered MVE of a seed, then rank each condition's five by research value."""
    lock = hold(seed_id)  # noqa: F841
    jobs, conditions = [], []
    for condition in CONDITIONS:
        folder = RUNS / seed_id / condition
        if not (folder / "second_commitment.json").exists():
            continue
        registrations = commitments(folder)[0]["content"]["registrations"]
        outcomes = [outcome(seed_id, condition, n) for n in range(1, 6)]
        conditions.append((condition, registrations, outcomes))
        for n, (registration, result) in enumerate(zip(registrations, outcomes), 1):
            if result and result["status"] == "delivered":
                archive = folder / f"mve{n}" / f"attempt{result['attempt']}" / "evaluation.tar.gz"
                jobs += [(folder / f"mve{n}" / f"audit_{auditor}.json",
                          lambda a=auditor, r=registration, o=result, x=archive, name=f"{condition}/{n}":
                          audit.audit_experiment(a, seed_id, r, o["implemented"], o["evaluated"], x, name))
                         for auditor in audit.AUDITORS]
    print(f"auditing: {len(jobs)} audits", flush=True)
    in_parallel(lambda job: step(*job), jobs, workers=8)

    for condition, registrations, outcomes in conditions:
        folder = RUNS / seed_id / condition
        experiments = []
        for n, (registration, result) in enumerate(zip(registrations, outcomes), 1):
            audits = [folder / f"mve{n}" / f"audit_{auditor}.json" for auditor in audit.AUDITORS]
            if result is None or result["status"] == "delivered" and not all(a.exists() for a in audits):
                break  # the condition is ranked only once all five are executed and audited
            delivered = result["status"] == "delivered"
            experiments.append({
                "experiment": n, "registration": registration, "delivered": delivered,
                "value": result["evaluated"]["value"] if delivered else None,
                "event_occurred": result["evaluated"]["event_occurred"] if delivered else None,
                "audits": [json.loads(a.read_text())["audit"] for a in audits] if delivered else []})
        if len(experiments) == 5:
            in_parallel(lambda auditor: step(folder / f"value_ranking_{auditor}.json", lambda: audit.rank_values(
                auditor, seed_id, condition, experiments)), audit.AUDITORS)
    status(seed_id)


def check_auditors(seed_id: str, count: int) -> None:
    """The check of the auditors: copies of audited MVEs, each with one known defect, are evaluated and
    audited like any other MVE. Reports how often each auditor flags a copy and the copy's own original."""
    lock = hold(seed_id)  # noqa: F841
    seed = json.loads((RUNS / seed_id / "seed.json").read_text())
    execute.stop_leftover_machines(seed_id)
    originals = []  # (condition, number, registration, outcome) of every MVE audited by all the auditors
    for condition in CONDITIONS:
        folder = RUNS / seed_id / condition
        if (folder / "second_commitment.json").exists():
            for n, registration in enumerate(commitments(folder)[0]["content"]["registrations"], 1):
                result = outcome(seed_id, condition, n)
                if result and result["status"] == "delivered" and all(
                        (folder / f"mve{n}" / f"audit_{a}.json").exists() for a in audit.AUDITORS):
                    originals.append((condition, n, registration, result))
    chosen = sorted(originals, key=lambda o: hashlib.sha256(f"{o[0]}/{o[1]}".encode()).hexdigest())[:count]

    def check(case: tuple[int, tuple]) -> None:
        index, (condition, n, registration, result) = case
        name = f"auditor-check/{condition}/{n}"
        folder = RUNS / seed_id / "auditor_check" / f"{condition}-mve{n}"
        attempt = RUNS / seed_id / condition / f"mve{n}" / f"attempt{result['attempt']}"
        planted = step(folder / "planted.json", lambda: audit.plant_defect(
            seed_id, registration, result["implemented"]["submission"], attempt / "submission.tar.gz",
            audit.DEFECTS[index % len(audit.DEFECTS)], folder / "submission.tar.gz", name))
        if not planted["planted"]:
            print(f"  no defect planted in {condition} MVE {n}: {planted['status']}", flush=True)
            return
        # The copy is audited with the planter's notes, so the original's notes do not give it away.
        implemented = {**result["implemented"], "submission": {**result["implemented"]["submission"],
                                                               "notes": planted["planted"].get("notes", "")}}
        evaluated = step(folder / "evaluation.json", lambda: execute.evaluate(
            seed, registration, implemented["submission"], folder, name))
        if evaluated["status"] != "delivered":
            print(f"  the copy of {condition} MVE {n} did not run: {evaluated['reason']}", flush=True)
            return
        for auditor in audit.AUDITORS:
            step(folder / f"audit_{auditor}.json", lambda: audit.audit_experiment(
                auditor, seed_id, registration, implemented, evaluated, folder / "evaluation.tar.gz", name))

    in_parallel(check, list(enumerate(chosen)), workers=PARALLEL_MVES)

    def judged(folder: Path, auditor: str) -> dict | None:
        path = folder / f"audit_{auditor}.json"
        return json.loads(path.read_text())["audit"] if path.exists() else None

    # Each copy is compared with its own original, auditor by auditor. A row says in how many of the
    # copies (or originals) an auditor found what the row names.
    rows: dict[str, dict[str, list[bool]]] = {}
    for condition, n, _, _ in chosen:
        folder = RUNS / seed_id / "auditor_check" / f"{condition}-mve{n}"
        planted = json.loads((folder / "planted.json").read_text()) if (folder / "planted.json").exists() else {}
        for auditor in audit.AUDITORS if planted.get("planted") else ():
            copy, original = judged(folder, auditor), judged(RUNS / seed_id / condition / f"mve{n}", auditor)
            if copy and original:
                defect = f"defect {audit.DEFECTS.index(planted['defect']) + 1}"
                found = {f"{defect}, copies flagged": audit.flagged(copy),
                         f"{defect}, originals flagged": audit.flagged(original),
                         "all copies flagged": audit.flagged(copy), "their originals flagged": audit.flagged(original),
                         "value 0, copies": copy["research_value"] == 0,
                         "value 0, originals": original["research_value"] == 0}
                for row, value in found.items():
                    rows.setdefault(row, {}).setdefault(auditor, []).append(bool(value))
    print("\nFlagged means judged unfaithful or invalid. The defects:")
    print("\n".join(f"  {number}. {defect}" for number, defect in enumerate(audit.DEFECTS, 1)))
    for row in sorted(rows):
        print(f"{row:28}", "   ".join(f"{auditor} {sum(found)}/{len(found)}" for auditor, found in rows[row].items()))


def review_seed(seed_id: str) -> None:
    """The review added after the study: each of the three auditor models reads the paper, one registration
    and what the auditors were shown of the delivered experiment, and answers the questions of
    prompts/review.md. Everything is saved under the seed's review/ folder; nothing else is touched."""
    lock = hold(seed_id)  # noqa: F841
    budget = Budget(REVIEW_CAP, spent(seed_id).get("review", 0.0))
    jobs = []
    for condition in CONDITIONS:
        folder = RUNS / seed_id / condition
        if not (folder / "second_commitment.json").exists():
            continue
        registrations = commitments(folder)[0]["content"]["registrations"]
        for n, (registration, result) in enumerate(zip(registrations, [outcome(seed_id, condition, n) for n in range(1, 6)]), 1):
            if result and result["status"] == "delivered":
                archive = folder / f"mve{n}" / f"attempt{result['attempt']}" / "evaluation.tar.gz"
                jobs += [(RUNS / seed_id / "review" / f"{condition}-mve{n}" / f"review_{reviewer}.json",
                          lambda a=reviewer, r=registration, o=result, x=archive, name=f"{condition}/{n}":
                          audit.review_experiment(a, seed_id, r, o["implemented"], o["evaluated"], x, name, budget))
                         for reviewer in audit.AUDITORS]
    todo = [job for job in jobs if not job[0].exists()]
    print(f"reviewing: {len(todo)} of {len(jobs)} reviews to make, ${budget.left:.2f} left of ${REVIEW_CAP:.0f}", flush=True)
    # One review of each reviewer goes first: a provider can serve the paper from its cache only after
    # one request has put it there. The others then run eight at a time, as the audits did.
    first = list({job[0].name: job for job in reversed(todo)}.values())
    in_parallel(lambda job: step(*job), first, workers=len(first) or 1)
    in_parallel(lambda job: budget.left > 0 and step(*job), [job for job in todo if job not in first], workers=8)
    status(seed_id)
    if budget.left <= 0 and not all(path.exists() for path, _ in jobs):
        sys.exit(f"The spending limit of ${REVIEW_CAP:.0f} for the reviews of seed {seed_id} is reached.")


def spent(seed_id: str) -> dict[str, float]:
    """What a seed has cost so far, by kind of step, from the spend log."""
    costs: dict[str, float] = {}
    for line in SPEND_LOG.read_text().splitlines() if SPEND_LOG.exists() else []:
        entry = json.loads(line)
        if entry["purpose"].startswith(seed_id + "/"):
            kind = entry["purpose"].split("/")[-1].split("-")[0]
            costs[kind] = costs.get(kind, 0.0) + entry["cost"]
    return costs


def status(seed_id: str) -> None:
    """Print, for each condition, how far it has come, and what the seed has cost so far."""
    folder = RUNS / seed_id
    for condition in CONDITIONS:
        if not (folder / condition / "proposal.json").exists():
            print(f"{condition:13} not proposed yet")
            continue
        proposal = json.loads((folder / condition / "proposal.json").read_text())
        if proposal["status"] != "done":
            print(f"{condition:13} ended while proposing: {proposal['status']} (${proposal['cost']:.2f})")
            continue
        saved = len(list((folder / condition).glob("*.json")))
        if not (folder / condition / "second_commitment.json").exists():
            print(f"{condition:13} proposing ${proposal['cost']:.2f}  NOT ready for execution ({saved} steps saved)")
            continue
        outcomes = [outcome(seed_id, condition, n) for n in range(1, 6)]
        delivered = sum(1 for o in outcomes if o and o["status"] == "delivered")
        failed = sum(1 for o in outcomes if o and o["status"] == "no delivery")
        audited = len(list((folder / condition).glob("mve*/audit_*.json"))) // len(audit.AUDITORS)
        ranked = len(list((folder / condition).glob("value_ranking_*.json")))
        print(f"{condition:13} proposing ${proposal['cost']:.2f}  committed; MVEs delivered {delivered}, "
              f"no delivery {failed}, to run {5 - delivered - failed}; audited {audited}; value rankings {ranked}")
    reviews = len(list((folder / "review").glob("*/review_*.json")))
    if reviews:
        print(f"reviews saved: {reviews}")
    costs = spent(seed_id)
    print("spent so far:", ", ".join(f"{kind} ${cost:.2f}" for kind, cost in sorted(costs.items())),
          f"| total ${sum(costs.values()):.2f}")


if __name__ == "__main__":
    command, seed_id, *rest = [word for word in sys.argv[1:] if word != "--new-code"]
    if command == "propose":
        assert all(c in CONDITIONS for c in rest), "conditions look like opus-single or luna-team"
        propose(seed_id, list(dict.fromkeys(rest)) or CONDITIONS)
    elif command == "execute":
        execute_seed(seed_id, int(rest[0]) if rest else 5)
    elif command == "audit":
        audit_seed(seed_id)
    elif command == "check-auditors":
        check_auditors(seed_id, int(rest[0]) if rest else 10)
    elif command == "review":
        review_seed(seed_id)
    else:
        status(seed_id)
    if FAILED:
        sys.exit(f"{len(FAILED)} tasks failed. Run the same command again; only what is missing is repeated.")
