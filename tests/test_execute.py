"""Starting machines, without any paid call."""

import threading
from types import SimpleNamespace

import pytest

from rtaste import execute

SEED = {"seed": "0000.00000", "code": "https://github.com/someone/repo", "code_commit": "abc"}


class FakeSandbox:
    id = "job"

    def run(self, command, **options):
        return SimpleNamespace(exit_code=0, timed_out=False, duration_ms=1)


@pytest.fixture
def quiet(monkeypatch):
    """No keys, no spend log, no pauses and no calls to Hugging Face."""
    monkeypatch.setenv("HF_READ_TOKEN", "x")
    monkeypatch.setenv("HF_COMPUTE_TOKEN", "x")
    pauses = []
    monkeypatch.setattr(execute, "load_env", lambda: None)
    monkeypatch.setattr(execute, "log_spend", lambda entry: None)
    monkeypatch.setattr(execute, "stop_machines", lambda labels: None)
    monkeypatch.setattr(execute, "_owner", lambda: "someone")
    monkeypatch.setattr(execute.time, "sleep", pauses.append)
    return pauses


def test_only_a_few_machines_start_at_the_same_moment(quiet, monkeypatch):
    starting, most, lock = [0], [0], threading.Lock()

    def create(**options):
        with lock:
            starting[0] += 1
            most[0] = max(most[0], starting[0])
        threading.Event().wait(0.05)
        with lock:
            starting[0] -= 1
        return FakeSandbox()

    monkeypatch.setattr(execute.Sandbox, "create", create)
    threads = [threading.Thread(target=execute.Machine(f"m{n}", 45, SEED).start) for n in range(12)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert most[0] == 4


def test_a_rate_limit_answer_is_waited_out_for_five_minutes(quiet, monkeypatch):
    answers = [429, 503, None]

    def create(**options):
        status = answers.pop(0)
        if status:
            error = RuntimeError("refused")
            error.response = SimpleNamespace(status_code=status)
            raise error
        return FakeSandbox()

    monkeypatch.setattr(execute.Sandbox, "create", create)
    machine = execute.Machine("m", 45, SEED)
    machine.start()
    assert quiet == [300, 60] and machine.sandbox is not None


def test_a_machine_that_cannot_start_is_a_machine_failure_and_a_bad_request_is_not_repeated(quiet, monkeypatch):
    calls = []

    def create(**options):
        calls.append(1)
        error = RuntimeError("no credit")
        error.response = SimpleNamespace(status_code=402)
        raise error

    monkeypatch.setattr(execute.Sandbox, "create", create)
    with pytest.raises(execute.MachineFailure):
        execute.Machine("m", 45, SEED).start()
    assert len(calls) == 1 and quiet == []


def test_waiting_for_a_machine_does_not_use_up_the_implementation_time(quiet, monkeypatch, tmp_path):
    clock = [1000.0]
    monkeypatch.setattr(execute.time, "time", lambda: clock[0])

    def create(**options):
        clock[0] += 40 * 60  # the machine takes forty minutes to arrive
        return FakeSandbox()

    monkeypatch.setattr(execute.Sandbox, "create", create)
    monkeypatch.setattr(execute.Machine, "save_workspace", lambda self, archive: 0)
    monkeypatch.setattr(execute.Machine, "stop", lambda self: 0.0)
    machine = execute.Machine("m", 600, SEED)  # a long machine limit, so only the implementation time can end it
    specs, tools = execute._executor_tools(machine, tmp_path, clock[0], {}, tmp_path / "submission.tar.gz")
    clock[0] += 10 * 60  # ten minutes of reading before the first command
    machine.start()
    assert machine.waited == 40 * 60
    clock[0] += 44 * 60  # 94 minutes after the first call, 54 of them the executor's own
    assert "time is over" not in tools["run"]({"command": "true", "timeout_minutes": 1})
    clock[0] += 37 * 60  # now 91 minutes of the executor's own time
    assert "time is over" in tools["run"]({"command": "true", "timeout_minutes": 1})
