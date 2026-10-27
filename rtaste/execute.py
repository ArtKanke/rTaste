"""Execution: an executor model implements one committed experiment on a GPU machine, and the
registered evaluation is then run on a fresh machine that the executor never touched.

Each MVE gets at most two attempts. If the first executor delivers nothing, the other executor
makes one independent attempt. A failure of the machines rather than of the executor is recorded
as such, and that attempt is repeated on the next run.
"""

import json
import os
import re
import shlex
import tarfile
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, Sandbox
from huggingface_hub.errors import SandboxError

from rtaste.models import Budget, Chat, load_env, log_spend, now, prompt, transient, work
from rtaste.predict import JUDGES
from rtaste.propose import Event
from rtaste.tools import READ_CHARS, paper_pages, paper_text, repo_ls, repo_read

EXECUTORS = ("opus", "sol")
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
FLAVOR = "h200"  # seeds 1 to 6 ran on "a100-large" at $2.50 an hour; seed 7 on "h200" (see data/README.md)
DOLLARS_PER_MINUTE = 5.00 / 60
EXECUTOR_TURNS = 60
EXECUTOR_SPEND = 6.0  # dollars for the executor's model calls
MACHINE_MINUTES = 45  # from the executor's first command
IMPLEMENTATION_MINUTES = 90  # for the whole implementation, model calls included
EVALUATION_MINUTES = 35
LARGEST_FILE = "200M"  # files above this size are not submitted (find -size syntax)
SUBMISSION_BYTES = 1_000_000_000
SHOWN_OUTPUT = 6_000  # characters of a command's output shown to the executor, from the end
# While a machine starts, the library asks Hugging Face for its state every two seconds, and an account
# may make 1,000 such requests in five minutes. Sixteen slow starts at once use that up, and every
# later start is then refused. So only a few machines start at the same moment.
STARTING = threading.BoundedSemaphore(4)
_owner_lock, _owner_name = threading.Lock(), []


def _owner() -> str:
    """The Hugging Face account the machines run under. It is asked for once, because that lookup is
    strictly rate limited and the library would otherwise repeat it for every start and every cancel."""
    with _owner_lock:
        if not _owner_name:
            load_env()
            _owner_name.append(HfApi().whoami(token=os.environ["HF_COMPUTE_TOKEN"])["name"])
        return _owner_name[0]


class MachineFailure(Exception):
    """The machine could not be started or stopped working. This is not the executor's failure."""


def _label(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", text)  # the characters Hugging Face allows in a label


class Machine:
    """One GPU sandbox. It starts when first used, runs for at most `minutes`, and is never restarted."""

    def __init__(self, purpose: str, minutes: float, seed: dict):
        self.purpose, self.minutes, self.seed = purpose, minutes, seed
        self.sandbox: Sandbox | None = None
        self.started = 0.0  # when the machine became ready
        self.waited = 0.0  # seconds between asking for the machine and having it
        self.stopped = False
        self.cost = 0.0
        self.failure = ""  # set when the machine itself failed; the attempt is then repeated
        self.pending: dict[str, str] = {}  # files written before the machine started

    def _fail(self, what: str, error: Exception | str) -> MachineFailure:
        self.failure = f"{what}: {error if isinstance(error, str) else f'{type(error).__name__}: {error}'}"
        return MachineFailure(self.failure)

    def start(self) -> None:
        if self.stopped or self.failure:
            raise MachineFailure(self.failure or "the machine has already stopped")
        load_env()
        asked = time.time()
        for attempt in range(3):
            try:
                with STARTING:
                    self.sandbox = Sandbox.create(
                        image=IMAGE, flavor=FLAVOR, idle_timeout=f"{int(self.minutes) + 15}m", start_timeout=900,
                        env={"HF_HOME": "/cache/huggingface"}, secrets={"HF_TOKEN": os.environ["HF_READ_TOKEN"]},
                        labels={"rtaste": _label(self.purpose), "rtaste-seed": _label(self.seed["seed"])},
                        namespace=_owner(), token=os.environ["HF_COMPUTE_TOKEN"])
                break
            except Exception as error:
                status = getattr(getattr(error, "response", None), "status_code", 0) or 0
                last = attempt == 2 or 400 <= status < 500 and status != 429  # a bad request is not repeated
                if not last:
                    time.sleep(300 if status == 429 else 60)  # a rate limit is counted over five minutes
                try:  # a machine that was created but never reached us would run, and cost, until its idle limit
                    stop_machines({"rtaste": self.purpose, "rtaste-seed": self.seed["seed"]})
                except Exception:
                    pass
                if last:
                    raise self._fail("could not start a machine", error) from error
        self.started = time.time()
        self.waited = self.started - asked
        log_spend({"time": now(), "model": FLAVOR, "purpose": self.purpose, "status": "machine started",
                   "job": self.sandbox.id, "waited": round(self.waited), "cost": 0.0})
        repo = self.seed["code"].split("github.com/")[1].strip("/")
        code = f"https://codeload.github.com/{repo}/tar.gz/{self.seed['code_commit']}"
        fetched = self.run(f"mkdir -p /workspace/seed && cd /workspace/seed && python -c \"import urllib.request; "
                           f"urllib.request.urlretrieve('{code}', '/tmp/seed.tar.gz')\" && tar xzf /tmp/seed.tar.gz "
                           "--strip-components=1", timeout=600)
        if fetched["exit_code"] != 0 or fetched["timed_out"]:
            raise self._fail("could not fetch the seed code", fetched["stderr"][-500:])
        for path, content in self.pending.items():
            self.write(path, content)
        self.pending = {}

    def seconds_left(self) -> float:
        if self.stopped:
            return 0.0
        return self.minutes * 60 - (time.time() - self.started if self.started else 0.0)

    def run(self, command: str, timeout: float) -> dict:
        """Run a command from /workspace. Only the end of its output is kept, however long it is."""
        if self.sandbox is None:
            self.start()
        tail = {"stdout": "", "stderr": ""}

        def keep(stream: str):
            return lambda text: tail.__setitem__(stream, (tail[stream] + text)[-2 * SHOWN_OUTPUT:])

        try:
            result = self.sandbox.run(command, cwd="/workspace", check=False, timeout=max(1.0, timeout),
                                      capture_output=False, on_stdout=keep("stdout"), on_stderr=keep("stderr"))
        except Exception as error:
            raise self._fail("the machine stopped working", error) from error
        return {"exit_code": result.exit_code, "timed_out": result.timed_out, "seconds": result.duration_ms / 1000,
                "stdout": tail["stdout"], "stderr": tail["stderr"]}

    def write(self, path: str, content: str) -> None:
        if self.sandbox is None:
            self.pending[path] = content
            return
        try:
            self.sandbox.run(f"mkdir -p {shlex.quote(str(Path(path).parent))}", cwd="/workspace", check=False)
            self.sandbox.files.write(path, content)
        except SandboxError as error:
            if (getattr(error, "status_code", None) or 500) < 500:
                raise RuntimeError(f"the machine refused to write {path}: {error}") from error  # the executor's doing
            raise self._fail("the machine stopped working", error) from error
        except Exception as error:
            raise self._fail("the machine stopped working", error) from error

    def read(self, path: str) -> str:
        if self.sandbox is None:
            return self.pending.get(path, f"No such file: {path}")
        try:
            return self.sandbox.files.read_text(path)
        except Exception as error:
            return f"Could not read {path}: {type(error).__name__}: {error}"

    def save_workspace(self, local: Path) -> int:
        """Copy /workspace to this computer as one archive. The seed code, installed packages, caches and
        files over the size limit are left out."""
        packed = self.run(f"find . -type f -not -path './seed/*' -not -path '*/site-packages/*' "
                          f"-not -path '*/__pycache__/*' -not -path '*/.cache/*' -not -size +{LARGEST_FILE} -print0 "
                          "| tar czf /tmp/workspace.tar.gz --null -T -", timeout=600)
        if packed["exit_code"] != 0:
            raise self._fail("could not pack the workspace", packed["stderr"][-500:])
        local.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):  # a dropped connection while copying is tried again
            try:
                self.sandbox.files.download("/tmp/workspace.tar.gz", local)
                break
            except Exception as error:
                if attempt == 2:
                    raise self._fail("could not copy the workspace", error) from error
                time.sleep(10)
        return local.stat().st_size

    def stop(self) -> float:
        """Stop the machine for good, record what it cost, and return its total cost."""
        if self.sandbox is not None:
            for _ in range(3):
                self.sandbox.kill()  # the library only logs a failure, and marks the machine killed on success
                if getattr(self.sandbox, "_killed", True):
                    break
                time.sleep(10)
            else:
                print(f"WARNING: could not stop machine {self.sandbox.id} ({self.purpose}); "
                      "it stops by itself once idle, or cancel it at huggingface.co/settings/jobs", flush=True)
            minutes = (time.time() - self.started) / 60 + 1  # billed by the minute, starting included
            self.cost += minutes * DOLLARS_PER_MINUTE
            log_spend({"time": now(), "model": FLAVOR, "purpose": self.purpose,
                       "status": "machine", "job": self.sandbox.id, "cost": round(minutes * DOLLARS_PER_MINUTE, 6),
                       "minutes": round(minutes, 2)})
            self.sandbox = None
        self.stopped = True
        return self.cost


def stop_machines(labels: dict[str, str]) -> None:
    """Cancel every machine that is running under these labels."""
    load_env()
    token, api, owner = os.environ["HF_COMPUTE_TOKEN"], HfApi(), _owner()
    labels = {name: _label(value) for name, value in labels.items()}
    for job in api.list_jobs(status=["RUNNING", "SCHEDULING"], labels=labels, namespace=owner, token=token):
        api.cancel_job(job_id=job.id, namespace=owner, token=token)
        print(f"stopped a machine that no run holds: {job.id} ({(job.labels or {}).get('rtaste')})", flush=True)


def stop_leftover_machines(seed_id: str) -> None:
    """Cancel machines of this seed that a stopped run left behind. Run before starting new ones."""
    stop_machines({"rtaste-seed": seed_id})


def _shown(output: str) -> str:
    return output if len(output) <= SHOWN_OUTPUT else "[earlier output cut]\n" + output[-SHOWN_OUTPUT:]


def _executor_tools(machine: Machine, repo: Path, started: float, submission: dict,
                    archive: Path) -> tuple[list[dict], dict]:
    """The executor's tools: read the seed code here, and write, read and run on the machine.

    When the machine's time or the implementation time is over, or the executor submits, the
    workspace is saved to `archive` and the machine is stopped at once, so idle time is not paid for.
    """

    def schema(properties: dict, required: list[str]) -> dict:
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}

    text, number = {"type": "string"}, {"type": "number"}

    def implementation_left() -> float:  # waiting for a machine is not the executor's time
        return IMPLEMENTATION_MINUTES * 60 - (time.time() - started - machine.waited)

    def finish_machine() -> None:
        if machine.sandbox is not None:
            try:
                machine.save_workspace(archive)
            finally:
                machine.stop()

    def closed() -> str | None:
        if machine.failure:
            return "The machine has failed. Stop now; this attempt will be repeated."
        if submission:
            return "You have submitted. Nothing can be changed now."
        if implementation_left() <= 0 or (machine.started and machine.seconds_left() <= 0):
            finish_machine()
            return "The time is over and the machine has stopped. Submit now, or stop."
        return None

    def at(path: str) -> str:
        return path if path.startswith("/") else f"/workspace/{path}"

    def write_file(args: dict) -> str:
        return closed() or (machine.write(at(args["path"]), args["content"]) or f"Wrote {at(args['path'])}.")

    def read_file(args: dict) -> str:
        if notice := closed():
            return notice
        content = machine.read(at(args["path"]))
        start = int(args.get("start", 0))
        return content[start:start + READ_CHARS] + (
            f"\n[more: call again with start={start + READ_CHARS}]" if len(content) > start + READ_CHARS else "")

    def run(args: dict) -> str:
        if notice := closed():
            return notice
        if machine.sandbox is None:
            machine.start()
        seconds = min(float(args.get("timeout_minutes", 10)) * 60, machine.seconds_left(), implementation_left())
        result = machine.run(args["command"], timeout=seconds)
        return (f"exit code {result['exit_code']}{' (stopped at the time limit)' if result['timed_out'] else ''}, "
                f"{result['seconds']:.0f} s. Machine time left: {machine.seconds_left() / 60:.0f} minutes.\n"
                f"--- stdout ---\n{_shown(result['stdout'])}\n--- stderr ---\n{_shown(result['stderr'])}")

    def submit(args: dict) -> str:
        if submission:
            return "You have already submitted."
        if not machine.started:
            return "Nothing has run yet. Test your code and your evaluation command before you submit."
        finish_machine()
        if machine.failure or not archive.exists():
            return "The machine has failed. Stop now; this attempt will be repeated."
        submission.update(evaluation_command=args["evaluation_command"], notes=args.get("notes", ""))
        return "Submitted. Reply with one line to finish."

    def seed_code(function):
        return lambda args: closed() or function(args)

    specs = [
        {"name": "repo_ls", "description": "List a directory of the seed paper's code (here, without the machine).",
         "parameters": schema({"path": text}, ["path"])},
        {"name": "repo_read", "description": f"Read a file of the seed paper's code, {READ_CHARS} characters at a time.",
         "parameters": schema({"path": text, "start": number}, ["path"])},
        {"name": "write_file", "description": "Write a text file on the machine. Paths are relative to /workspace.",
         "parameters": schema({"path": text, "content": text}, ["path", "content"])},
        {"name": "read_file", "description": f"Read a text file on the machine, {READ_CHARS} characters at a time.",
         "parameters": schema({"path": text, "start": number}, ["path"])},
        {"name": "run", "description": "Run a shell command on the machine from /workspace and wait for it. The first "
                                       "call starts the machine. Output is cut to its last part.",
         "parameters": schema({"command": text, "timeout_minutes": number}, ["command", "timeout_minutes"])},
        {"name": "submit", "description": "Submit once: the command that runs the registered evaluation on a fresh "
                                          "machine, and short notes on what you implemented and any change you made.",
         "parameters": schema({"evaluation_command": text, "notes": text}, ["evaluation_command", "notes"])},
    ]
    tools = {"repo_ls": seed_code(lambda a: repo_ls(repo, a.get("path", ""))),
             "repo_read": seed_code(lambda a: repo_read(repo, a["path"], int(a.get("start", 0)))),
             "write_file": write_file, "read_file": read_file, "run": run, "submit": submit}
    return specs, tools


def implement(executor: str, seed: dict, registration: dict, repo: Path, folder: Path, purpose: str) -> dict:
    """One executor's attempt: implement the experiment and submit. Returns the attempt record.

    A failure of the machine or of the provider's service is raised, not recorded, so that the
    attempt is repeated on the next run.
    """
    machine = Machine(purpose + "/machine", MACHINE_MINUTES, seed)
    submission: dict = {}
    started = time.time()
    specs, tools = _executor_tools(machine, repo, started, submission, folder / "submission.tar.gz")
    chat = Chat(executor, prompt("system"), specs, effort="high", purpose=purpose + "/executor",
                budget=Budget(EXECUTOR_SPEND))
    task = prompt("executor", arxiv_id=seed["seed"], paper=paper_text(seed["seed"]),
                  registration=json.dumps(registration, indent=1), machine_minutes=str(MACHINE_MINUTES),
                  turns=str(EXECUTOR_TURNS), spend=f"{EXECUTOR_SPEND:.0f}", implementation_minutes=str(IMPLEMENTATION_MINUTES),
                  evaluation_minutes=str(EVALUATION_MINUTES),
                  raw_outputs=", ".join(registration["evaluation"]["raw_outputs"]))
    try:
        reply = work(chat, task, tools, EXECUTOR_TURNS, images=paper_pages(seed["seed"]))
        why = f"did not submit ({reply.status}): {reply.text[-500:]}"
    except Exception as error:  # an exhausted budget or an invalid request ends the attempt
        account = getattr(error, "status_code", None) in (401, 402, 403) or "credit balance" in str(error).lower()
        if (transient(error) or account) and not submission:
            raise  # a service or account problem: nothing is saved and the attempt is made again
        why = f"{type(error).__name__}: {error}"
    finally:
        compute_cost = machine.stop()  # usually stopped already, at submission
    if machine.failure:  # tool errors reach the model, not us, so the machine says what happened
        raise MachineFailure(machine.failure)
    size = (folder / "submission.tar.gz").stat().st_size if submission else 0
    if not submission:
        status, reason = "no delivery", why
    elif size > SUBMISSION_BYTES:
        status, reason = "no delivery", f"the submission is {size / 1e9:.1f} GB, above the 1 GB limit"
    else:
        status, reason = "submitted", ""
    return {"executor": executor, "status": status, "reason": reason, "submission": submission,
            "model_cost": chat.cost, "compute_cost": compute_cost, "minutes": (time.time() - started) / 60,
            "transcript": chat.transcript}


def check_result(text: str) -> tuple[float | None, str]:
    """Read result.json. Returns the value, or None and what is wrong."""
    try:
        value = json.loads(text)["value"]
    except (json.JSONDecodeError, KeyError, TypeError):
        value = None
    if isinstance(value, bool) or not isinstance(value, (int, float)):  # not true, not "0.5"
        return None, "result.json is missing or is not {\"statistic\": ..., \"value\": <a number>}"
    if value != value or value in (float("inf"), float("-inf")):
        return None, "the value in result.json is not a finite number"
    return float(value), ""


def _inside_workspace(path: str) -> bool:
    clean = os.path.normpath(path)
    return bool(path) and not os.path.isabs(clean) and clean not in (".", "seed") and not clean.startswith(("..", "seed/"))


def evaluate(seed: dict, registration: dict, submission: dict, folder: Path, purpose: str) -> dict:
    """The protected evaluation: run the submitted evaluation command on a fresh machine.

    The results the executor left from its own tests are deleted first, so whatever is recorded
    comes from this run. A machine failure is raised, not recorded, so that the evaluation is
    repeated on the next run.
    """
    machine = Machine(purpose + "/evaluation", EVALUATION_MINUTES + 10, seed)
    record = {"status": "", "reason": "", "started_at": now()}
    stale = ["result.json"] + [p for p in registration["evaluation"]["raw_outputs"] if _inside_workspace(p)]
    # What the executor's own last run gave, kept next to the protected result to show how much a rerun moves it.
    with tarfile.open(folder / "submission.tar.gz") as submitted:
        own = next((submitted.extractfile(m).read().decode("utf-8", "replace") for m in submitted.getmembers()
                    if m.name in ("./result.json", "result.json")), "")
    record["executor_value"] = check_result(own)[0]
    try:
        machine.start()
        for attempt in range(3):  # a dropped connection while copying is tried again
            try:
                machine.sandbox.files.upload(folder / "submission.tar.gz", "/tmp/submission.tar.gz")
                break
            except Exception as error:
                if attempt == 2:
                    raise machine._fail("could not copy the submission", error) from error
                time.sleep(10)
        unpacked = machine.run("tar xzf /tmp/submission.tar.gz -C /workspace && rm -rf -- "
                               + " ".join(shlex.quote(p) for p in stale) + " && touch /tmp/evaluation-started",
                               timeout=600)
        if unpacked["exit_code"] != 0:
            raise machine._fail("could not unpack the submission", unpacked["stderr"][-500:])
        result = machine.run(submission["evaluation_command"], timeout=EVALUATION_MINUTES * 60)
        record["command"] = {k: (_shown(v) if isinstance(v, str) else v) for k, v in result.items()}
        machine.run("find . -type f -newer /tmp/evaluation-started -not -path './seed/*' -not -path '*/site-packages/*' "
                    "| head -5000 > /tmp/written.txt", timeout=120)
        machine.save_workspace(folder / "evaluation.tar.gz")
        try:
            listed = machine.sandbox.files.read_text("/tmp/written.txt").splitlines()
            record["written_by_evaluation"] = [line.removeprefix("./") for line in listed]
            written = machine.sandbox.files.exists("/workspace/result.json")
            value, problem = check_result(machine.sandbox.files.read_text("/workspace/result.json") if written else "")
        except Exception as error:
            raise machine._fail("could not read the evaluation's results", error) from error
        missing = [p for p in stale[1:] if machine.run(f"test -e {shlex.quote(p)}", timeout=30)["exit_code"] != 0]
        if result["exit_code"] != 0 or result["timed_out"]:
            record["status"], record["reason"] = "no delivery", (
                "the evaluation did not finish in time" if result["timed_out"] else f"exit code {result['exit_code']}")
        elif value is None:
            record["status"], record["reason"] = "no delivery", problem
        else:
            record["status"], record["value"] = "delivered", value
            record["event_occurred"] = Event(**registration["event"]).occurred(value)
            record["missing_raw_outputs"] = missing
    finally:
        record["compute_cost"] = machine.stop()
    return record


def executor_for(model: str, seed_position: int) -> str:
    """The first executor for a proposing model's MVEs on a seed: the two alternate across seeds,
    and neighbouring models start with different executors."""
    return EXECUTORS[(seed_position + JUDGES.index(model)) % 2]
