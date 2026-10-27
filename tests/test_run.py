"""Saving and resuming steps, without any paid call."""

import json

import pytest

from rtaste import run
from rtaste.execute import check_result
from rtaste.predict import commit, second_commitment
from rtaste.run import in_parallel, step


def test_a_finished_step_is_saved_and_not_carried_out_again(tmp_path):
    calls = []

    def make():
        calls.append(1)
        return {"value": len(calls)}

    assert step(tmp_path / "a" / "step.json", make) == {"value": 1}
    assert step(tmp_path / "a" / "step.json", make) == {"value": 1}
    assert len(calls) == 1


def test_a_step_that_fails_saves_nothing_and_does_not_stop_the_others(tmp_path, capsys):
    def task(number):
        return step(tmp_path / f"{number}.json", lambda: {"n": 10 // number})

    assert in_parallel(task, [1, 0, 5]) == [{"n": 10}, None, {"n": 2}]
    assert not (tmp_path / "0.json").exists() and "FAILED 0" in capsys.readouterr().out
    assert run.FAILED == ["0"]  # the command then ends with an error
    run.FAILED.clear()


def test_commitments_are_checked_before_they_are_used(tmp_path):
    first = commit({"registrations": [{"title": "a"}]})
    second = second_commitment(first, {"predictions": {"ranking": [1]}}, [{"judge": "luna", "predictions": None}])
    (tmp_path / "first_commitment.json").write_text(json.dumps(first))
    (tmp_path / "second_commitment.json").write_text(json.dumps(second))
    assert run.commitments(tmp_path) == (first, second)
    first["content"]["registrations"][0]["title"] = "b"  # an experiment edited after it was committed
    (tmp_path / "first_commitment.json").write_text(json.dumps(first))
    with pytest.raises(ValueError):
        run.commitments(tmp_path)
    (tmp_path / "first_commitment.json").write_text(json.dumps(commit(first["content"])))  # hashed again
    with pytest.raises(ValueError):  # the second commitment still names the first as it was
        run.commitments(tmp_path)


def test_a_seed_is_not_continued_with_other_code_or_another_paper(tmp_path, monkeypatch):
    seed = {"seed": "s", "code_version": "a" * 40, "paper_sha256": "p", "pdf_sha256": "q", "pages": 3}
    monkeypatch.setattr(run, "RUNS", tmp_path)
    (tmp_path / "s").mkdir()
    monkeypatch.setattr(run, "seed_inputs", lambda seed_id: {"paper_sha256": "p", "pdf_sha256": "q", "pages": 3})
    monkeypatch.setattr(run, "code_version", lambda: "a" * 40)
    run.check_unchanged(seed)
    monkeypatch.setattr(run, "code_version", lambda: "b" * 40)
    with pytest.raises(SystemExit):
        run.check_unchanged(seed)
    monkeypatch.setattr(run.sys, "argv", ["run", "audit", "s", "--new-code"])
    run.check_unchanged(seed)  # accepted, and written down
    assert json.loads((tmp_path / "s" / "code_changes.jsonl").read_text())["code_version"] == "b" * 40
    monkeypatch.setattr(run.sys, "argv", ["run", "audit", "s"])
    run.check_unchanged(seed)  # the recorded version is now the seed's version
    monkeypatch.setattr(run, "seed_inputs", lambda seed_id: {"paper_sha256": "other", "pdf_sha256": "q", "pages": 3})
    with pytest.raises(SystemExit):
        run.check_unchanged(seed)


def test_a_result_must_be_a_finite_number():
    assert check_result('{"statistic": "s", "value": 0.5}') == (0.5, "")
    assert check_result('{"statistic": "s", "value": 3}') == (3.0, "")
    for wrong in ('{"value": true}', '{"value": "0.5"}', '{"value": NaN}', '{"value": Infinity}', '{"value": null}',
                  '{"statistic": "s"}', "not json", "[1]"):
        assert check_result(wrong)[0] is None, wrong


def test_the_review_goes_over_every_delivered_mve_once_for_each_reviewer(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RUNS", tmp_path)
    monkeypatch.setattr(run, "SPEND_LOG", tmp_path / "spend.jsonl")
    monkeypatch.setattr(run, "CONDITIONS", ["opus-single", "luna-team"])
    for condition in run.CONDITIONS:
        (tmp_path / "s" / condition).mkdir(parents=True)
        (tmp_path / "s" / condition / "second_commitment.json").write_text("{}")
    monkeypatch.setattr(run, "commitments", lambda folder: ({"content": {"registrations": [{"title": n} for n in range(5)]}}, {}))
    delivered = {"status": "delivered", "attempt": 1, "implemented": {}, "evaluated": {}}
    monkeypatch.setattr(run, "outcome", lambda seed_id, condition, n: delivered if n != 4 else {"status": "no delivery"})
    calls = []

    def review(reviewer, seed_id, registration, implemented, evaluated, archive, name, budget):
        calls.append((reviewer, name, archive.relative_to(tmp_path).as_posix()))
        budget.add(1.0)
        with (tmp_path / "spend.jsonl").open("a") as log:
            log.write(json.dumps({"purpose": f"{seed_id}/{name}/review-{reviewer}", "cost": 1.0}) + "\n")
        return {"reviewer": reviewer, "status": "done"}

    monkeypatch.setattr(run.audit, "review_experiment", review)
    monkeypatch.setattr(run, "status", lambda seed_id: None)
    batches, in_parallel = [], run.in_parallel

    def recorded(function, items, workers=10):
        batches.append(([job[0].name for job in items], len(calls)))
        return in_parallel(function, items, workers)

    monkeypatch.setattr(run, "in_parallel", recorded)
    run.review_seed("s")
    # One review of each reviewer is finished before the others start, so that the paper is in the cache.
    assert sorted(batches[0][0]) == ["review_grok.json", "review_opus.json", "review_sol.json"] and batches[1][1] == 3
    assert len(calls) == 2 * 4 * 3 and not any("/4" in name for _, name, _ in calls)  # the undelivered MVE has no review
    assert sorted(name for reviewer, name, _ in calls if reviewer == "grok" and name.startswith("opus-single")) == [
        f"opus-single/{n}" for n in (1, 2, 3, 5)]
    assert ("opus", "luna-team/1", "s/luna-team/mve1/attempt1/evaluation.tar.gz") in calls
    saved = sorted(path.relative_to(tmp_path / "s" / "review").as_posix() for path in (tmp_path / "s" / "review").rglob("*.json"))
    assert len(saved) == 24 and "luna-team-mve1/review_sol.json" in saved
    assert {reviewer for reviewer, _, _ in calls[:3]} == set(run.audit.AUDITORS)  # one of each goes first, for the cache
    run.review_seed("s")
    assert len(calls) == 24  # nothing is reviewed a second time

    monkeypatch.setattr(run, "REVIEW_CAP", 25.0)  # the limit holds over every run of the stage: 24 dollars are spent
    (tmp_path / "s" / "review" / "luna-team-mve1" / "review_sol.json").unlink()
    (tmp_path / "s" / "review" / "luna-team-mve2" / "review_sol.json").unlink()
    (tmp_path / "s" / "review" / "luna-team-mve2" / "review_opus.json").unlink()
    with pytest.raises(SystemExit):
        run.review_seed("s")
    assert 25 <= len(calls) <= 26  # it stops as soon as the limit is reached
    monkeypatch.setattr(run, "REVIEW_CAP", 40.0)
    run.review_seed("s")
    monkeypatch.setattr(run, "REVIEW_CAP", 1.0)
    run.review_seed("s")  # everything is saved, so a limit that was passed is no longer an error

