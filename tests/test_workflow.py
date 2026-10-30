"""Small end-to-end checks for the template, not method certification."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import optuna
import pytest
import torch
from filelock import Timeout
from pydantic import ValidationError
from typer.testing import CliRunner

from core import analysis, groups, plotting, runner, tuning
from core.cli import app
from core.config import ROOT, Config, Pruning, changed, dumps, load
from core.platforms import PlatformError
from core.records import Run, atomic, checkpoint, history, locked, read_json
from core.training import Evaluation, evaluate
from methods import linear, ragged


class TrialFailure(Exception):
    pass


def cfg(root, method="linear", **updates):
    base = load(ROOT / "configs" / f"{method}.toml").model_dump()
    base.update(updates)
    return Config.model_validate(base)


def test_config_cli_and_protection(tmp_path):
    c = cfg(tmp_path)
    atomic(tmp_path / "input.toml", dumps(c))
    actual = load("input.toml", ["params.lr=0.02", 'device="cpu"'], tmp_path)
    assert actual.params["lr"] == 0.02
    assert c.search.pruning.model_dump() == {"strategy": "none", "params": {}}
    assert cfg(tmp_path, search={}).search.pruning == Pruning()
    median = load(
        "input.toml",
        ['search.pruning.strategy="median"', "search.pruning.params.n_warmup_steps=2"],
        tmp_path,
    )
    assert median.search.pruning.params == {
        "n_startup_trials": 5,
        "n_warmup_steps": 2,
        "interval_steps": 1,
        "n_min_trials": 1,
    }
    for invalid in [
        ["search.pruning=true"],
        ['search.pruning.strategy="hyperband"'],
        ["search.pruning.params.n_warmup_steps=1"],
        ['search.pruning.strategy="median"', "search.pruning.params.typo=1"],
        ['search.pruning.strategy="median"', "search.pruning.params.interval_steps=0"],
        [
            'search.pruning.strategy="median"',
            "search.pruning.params.n_startup_trials=-1",
        ],
        [
            'search.pruning.strategy="median"',
            'search.pruning.params.n_warmup_steps="2"',
        ],
    ]:
        with pytest.raises(ValidationError):
            load("input.toml", invalid, tmp_path)
    with pytest.raises(ValidationError):
        load("input.toml", ['train.epochs="4"'], tmp_path)
    with pytest.raises(ValidationError):
        load("input.toml", ["train.epohs=4"], tmp_path)
    with pytest.raises(ValueError):
        changed(c, actual, {"device"})
    for key in ("seed", "data"):
        values = c.model_dump()
        del values[key]
        with pytest.raises(ValidationError):
            Config.model_validate(values)
    optional = Config.model_validate(
        c.model_dump() | {"data": {"name": "memory"}, "train": {}}
    )
    assert optional.data.path is None and optional.train.metric is None
    with pytest.raises(ValueError, match="train.epochs"):
        runner.execute(optional, root=tmp_path)
    cli = CliRunner()
    assert cli.invoke(app, ["tune", "input.toml"]).exit_code == 2
    assert (
        cli.invoke(app, ["paired", "accuracy", "--left", "a", "--right", "b"]).exit_code
        == 2
    )
    assert (
        cli.invoke(
            app, ["plot", "one", "--metric", "loss", "--destination", "a.png"]
        ).exit_code
        == 2
    )
    assert cli.invoke(app, ["group", "--help"]).exit_code == 0
    assert cli.invoke(app, ["group-create", "--help"]).exit_code == 2
    assert (
        cli.invoke(app, ["prepare", "input.toml", "--root", str(tmp_path)]).exit_code
        == 0
    )
    result = cli.invoke(
        app, ["train", "input.toml", "--root", str(tmp_path), "--id", "one"]
    )
    assert result.exit_code == 0, result.output
    target = tmp_path / "experiments/runs/one"
    original = (target / "record.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.execute(c, id="one", root=tmp_path)
    assert (target / "record.json").read_bytes() == original
    with pytest.raises(ValueError):
        runner.execute(c, id="../one", root=tmp_path)
    with locked(target), pytest.raises(Timeout):
        runner.execute(c, id="one", overwrite=True, root=tmp_path)
    # A separate process owns the OS lock; abrupt os._exit releases it without cleanup.
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys,os; from pathlib import Path; from core.records import locked;\nwith locked(Path(sys.argv[1])):\n print('locked', flush=True); input(); os._exit(7)",
            str(target),
        ],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(Timeout), locked(target):
            pass
    finally:
        child.stdin.write("\n")
        child.stdin.flush()
        child.wait(timeout=10)
    with locked(target):
        pass
    runner.execute(c, id="one", overwrite=True, root=tmp_path)
    assert len(read_json(target / "record.json")["launches"]) == 1
    # A test cannot overwrite a training run, even with explicit overwrite.
    with pytest.raises(ValueError):
        runner.test(c, source="one", id="one", overwrite=True, root=tmp_path)
    runner.execute(c, id="two", root=tmp_path)
    with pytest.raises(ValueError):
        runner.test(c, source="one", id="two", overwrite=True, root=tmp_path)


@pytest.mark.parametrize("method,suffix", [("linear", "pt"), ("ragged", "json")])
def test_prepare_overwrite(tmp_path, method, suffix):
    c = cfg(tmp_path, method)
    target = tmp_path / c.data.path
    target.mkdir(parents=True)
    assert runner.prepare(c, tmp_path) == target
    outputs = [
        target / f"{split}.{suffix}" for split in ("train", "validation", "test")
    ]
    note = target / "README.txt"
    note.write_text("Dataset notes", encoding="utf-8")
    outputs[0].unlink()
    for file in outputs[1:]:
        file.write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        runner.prepare(c, tmp_path)
    assert not outputs[0].exists()
    assert all(file.read_bytes() == b"existing data" for file in outputs[1:])

    assert runner.prepare(c, tmp_path, overwrite=True) == target
    for file, size in zip(outputs, (48, 20, 20), strict=True):
        if suffix == "json":
            assert len(read_json(file)) == size
        else:
            assert len(torch.load(file, weights_only=True)["y"]) == size

    atomic(tmp_path / "input.toml", dumps(c))
    args = ["prepare", "input.toml", "--root", str(tmp_path)]
    cli = CliRunner()
    result = cli.invoke(app, args)
    assert isinstance(result.exception, FileExistsError)
    outputs[0].write_bytes(b"replace me")
    result = cli.invoke(app, args + ["--overwrite"])
    assert result.exit_code == 0, result.output
    assert "Prepared data:" in result.output
    assert outputs[0].read_bytes() != b"replace me"
    assert note.read_text(encoding="utf-8") == "Dataset notes"
    assert set(target.iterdir()) == {*outputs, note}


def test_prepare_optional_overwrite(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    calls = []

    def prepare(config, root):
        calls.append((config, root))
        return root

    monkeypatch.setattr(linear, "prepare", prepare)
    assert runner.prepare(c, tmp_path) == tmp_path
    with pytest.raises(TypeError, match="overwrite"):
        runner.prepare(c, tmp_path, overwrite=True)
    assert calls == [(c, tmp_path)]


def test_default_resume_lineage_and_independent_test(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    runner.prepare(c, tmp_path)
    runner.execute(c, id="continuous", root=tmp_path)
    original_log = Run.log

    def interrupt(self, values, epoch, split="validation"):
        original_log(self, values, epoch, split)
        if epoch == 2 and split == "validation":
            raise TrialFailure("after metrics but before checkpoint")

    monkeypatch.setattr(Run, "log", interrupt)
    with pytest.raises(TrialFailure):
        runner.execute(c, id="interrupted", root=tmp_path)
    target = tmp_path / "experiments/runs/interrupted"
    assert {row["epoch"] for row in history(target)} == {1}
    failed_record = (target / "record.json").read_bytes()
    runner.test(c, source="interrupted", id="failed-source-test", root=tmp_path)
    assert (target / "record.json").read_bytes() == failed_record
    with (target / "metrics-1.jsonl").open("a") as file:
        file.write('{"launch":')  # Interrupted JSONL append, retained for diagnosis.
    monkeypatch.setattr(Run, "log", original_log)
    runner.execute(c, id="interrupted", resume=True, root=tmp_path)
    actual = torch.load(checkpoint(target, "last"), weights_only=True)
    expected = torch.load(
        checkpoint(tmp_path / "experiments/runs/continuous", "last"), weights_only=True
    )
    for key, value in expected["model"].items():
        assert torch.equal(value, actual["model"][key])
    assert actual["scheduler"] == expected["scheduler"]
    assert actual["early_stop"] == expected["early_stop"]
    assert [
        (r["epoch"], r["launch"]) for r in history(target) if r["split"] == "validation"
    ] == [(1, 1), (2, 2), (3, 2), (4, 2)]
    assert (
        sum(len(f.read_text().splitlines()) for f in target.glob("metrics-*.jsonl"))
        == 11
    )
    management = read_json(target / "checkpoint.json")
    assert {f.name for f in target.glob("checkpoint-*.pt")} == {management["file"]} | {
        item["file"] for item in management["top"]
    }
    before = (target / "record.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(
            Run, "open_platforms", lambda self: pytest.fail("No-op opened platforms")
        )
        runner.execute(c, id="interrupted", resume=True, root=tmp_path)
    assert (target / "record.json").read_bytes() == before
    assert "step" not in management
    assert all("step" not in row for row in history(target))
    early = c.model_copy(
        update={
            "train": c.train.model_copy(
                update={"epochs": 6, "patience": 1, "min_delta": 2.0, "keep_top_k": 2}
            )
        }
    )
    runner.execute(early, id="early", root=tmp_path)
    runner.execute(early, id="early", resume=True, root=tmp_path)
    stopped = tmp_path / "experiments/runs/early"
    assert read_json(stopped / "checkpoint.json")["epoch"] == 2
    assert len(list(stopped.glob("checkpoint-*.pt"))) == 2
    # Test construction cannot read train/validation datasets.
    data = tmp_path / c.data.path
    (data / "train.pt").unlink()
    (data / "validation.pt").unlink()
    runner.test(c, source="interrupted", id="evaluation", root=tmp_path)
    assert (target / "record.json").read_bytes() == before
    result = tmp_path / "experiments/runs/evaluation"
    assert (result / "predictions.pt").exists()
    runner.test(c, external=checkpoint(target), id="external", root=tmp_path)
    runner.test(c, source="interrupted", id="evaluation", overwrite=True, root=tmp_path)
    assert len(read_json(result / "record.json")["launches"]) == 1
    bad = c.model_copy(update={"params": c.params | {"hidden": 4}})
    with pytest.raises(ValueError):
        runner.test(bad, source="interrupted", root=tmp_path)


def test_weighted_evaluation_and_custom_loop(tmp_path):
    model = torch.nn.Identity()
    data = [(torch.tensor([[1.0], [2.0]]), 3.0, 2), (torch.tensor([[3.0]]), 9.0, 1)]

    def batch(model, row, device):
        x, loss, weight = row
        return {
            "loss": torch.tensor(loss),
            "weight": weight,
            "predictions": x,
            "targets": x,
        }

    parts = Evaluation(model, data, batch, lambda p, t: {"sum": p.sum().item()})
    values, p, t = evaluate(parts)
    assert values == {"sum": 6.0, "loss": 5.0}
    assert p.shape == t.shape == (3, 1)
    with pytest.raises(ValueError):
        evaluate(replace(parts, data=[]))
    c = cfg(tmp_path, "ragged")
    runner.prepare(c, tmp_path)
    runner.execute(c, id="custom", root=tmp_path)
    target = tmp_path / "experiments/runs/custom"
    assert ragged.load_state(checkpoint(target, "last"))["memory"].epoch == 3
    extended = c.model_copy(update={"train": c.train.model_copy(update={"epochs": 4})})
    runner.execute(extended, id="custom", resume=True, root=tmp_path)
    assert ragged.load_state(checkpoint(target, "last"))["memory"].epoch == 4
    runner.test(c, source="custom", id="custom-test", root=tmp_path)
    runner.test(
        c, external=checkpoint(target, "last"), id="custom-external", root=tmp_path
    )
    assert (
        read_json(tmp_path / "experiments/runs/custom-external/record.json")["result"][
            "mse"
        ]
        < 0.01
    )
    assert (
        read_json(tmp_path / "experiments/runs/custom-test/record.json")["result"][
            "mse"
        ]
        < 0.01
    )


def test_ragged_search(tmp_path):
    values, _ = ragged.evaluate(
        torch.nn.Identity(), [{"observations": [0.0], "target": 2.0}], "cpu"
    )
    assert values == {"mse": 4.0}
    c = cfg(tmp_path, "ragged")
    runner.prepare(c, tmp_path)
    for optimizer, hidden, count in (("sgd", 0, 2), ("adam", 4, 13)):
        params = {
            "params.lr": 0.02,
            "params.optimizer": optimizer,
            "params.hidden": hidden,
        }
        if optimizer == "sgd":
            params["params.momentum"] = 0.5
        actual = tuning.sample(c, optuna.trial.FixedTrial(params))
        assert actual.params["optimizer"] == optimizer
        assert ("momentum" in actual.params) == (optimizer == "sgd")
        assert c.params["momentum"] == 0.0
        runner.execute(actual, id=optimizer, root=tmp_path)
        target = tmp_path / "experiments/runs" / optimizer
        final = read_json(target / "record.json")["result"]
        assert sum(p.numel() for p in ragged.network(actual).parameters()) == count
        rows = [row for row in history(target) if row["split"] == "validation"]
        assert final == rows[-1]["metrics"]
        saved = read_json(target / "checkpoint.json")
        best_mse = min(row["metrics"]["mse"] for row in rows)
        assert saved["top"][0]["score"] == best_mse
        assert ragged.objective(actual, target, None) == best_mse + 0.001 * count

    tuning.tune(c, trials=1, id="custom-search", root=tmp_path)
    study = optuna.load_study(
        study_name="custom-search",
        storage=tuning.storage(tmp_path / "experiments/studies/custom-search"),
    )
    trial = study.trials[0]
    target = tmp_path / "experiments/runs" / trial.user_attrs["run_id"]
    rows = [row for row in history(target) if row["split"] == "validation"]
    count = 2 if trial.params["params.hidden"] == 0 else 13
    expected = min(row["metrics"]["mse"] for row in rows) + 0.001 * count
    assert trial.value == expected
    assert trial.intermediate_values == {
        row["epoch"]: row["metrics"]["mse"] for row in rows
    }
    atomic(target / "record.json", {"result": {"mse": 123.0}})
    assert ragged.objective(c, target, None) == expected


def test_optuna_fail_retry_continue_prune_export(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    c.search.catch = ["tests.test_workflow.TrialFailure"]
    pruners = []
    create_study = optuna.create_study

    def capture_pruner(**kwargs):
        pruners.append(kwargs["pruner"])
        return create_study(**kwargs)

    monkeypatch.setattr(optuna, "create_study", capture_pruner)
    runner.prepare(c, tmp_path)
    original = linear.build
    calls = 0

    def flaky(config, root):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TrialFailure("explicitly tolerated")
        return original(config, root)

    monkeypatch.setattr(linear, "build", flaky)
    tuning.tune(c, trials=2, id="search", root=tmp_path)
    target = tmp_path / "experiments/studies/search"
    study = optuna.load_study(study_name="search", storage=tuning.storage(target))
    assert isinstance(pruners[0], optuna.pruners.NopPruner)
    assert [t.state.name for t in study.trials] == ["FAIL", "COMPLETE"]
    assert list(study.trials[1].intermediate_values) == list(
        range(1, c.train.epochs + 1)
    )
    assert load(target / "config.toml").search.pruning.strategy == "none"
    failed_run = tmp_path / "experiments/runs" / study.trials[0].user_attrs["run_id"]
    assert read_json(failed_run / "record.json")["status"] == "failed"
    before = (target / "record.json").read_bytes()
    with pytest.raises(ValueError):
        tuning.tune(c, trials=1, id="search", continuation=True, retry=1, root=tmp_path)
    assert (target / "record.json").read_bytes() == before
    tuning.tune(c, trials=1, id="search", continuation=True, retry=0, root=tmp_path)
    assert study.trials[2].params == study.trials[0].params
    assert study.trials[2].user_attrs["retry_of"] == 0
    exported = tuning.export(c, "search", "selected.toml", root=tmp_path)
    selected = load(exported, root=tmp_path)
    assert selected.provenance["study"] == "search"
    with pytest.raises(FileExistsError):
        tuning.export(c, "search", "selected.toml", root=tmp_path)
    tuning.export(c, "search", "selected.toml", overwrite=True, root=tmp_path)
    assert selected.params["lr"] == study.best_trial.params["params.lr"]
    with pytest.raises(ValueError):
        tuning.tune(
            c.model_copy(update={"seed": 101}),
            trials=1,
            id="search",
            continuation=True,
            root=tmp_path,
        )
    with pytest.raises(ValueError):
        tuning.exceptions(["builtins.RuntimeError"])
    with pytest.raises(ValueError):
        tuning.exceptions(["core.platforms.PlatformError"])

    def fatal(config, root):
        raise RuntimeError("not tolerated")

    monkeypatch.setattr(linear, "build", fatal)
    with pytest.raises(RuntimeError):
        tuning.tune(c, trials=2, id="fatal", root=tmp_path)
    assert (
        read_json(tmp_path / "experiments/studies/fatal/record.json")["status"]
        == "failed"
    )
    monkeypatch.setattr(linear, "build", original)
    c.search.pruning = Pruning(
        strategy="median", params={"n_startup_trials": 1, "n_warmup_steps": 2}
    )
    scores = iter([1.0] * c.train.epochs + [0.0, 0.0])
    monkeypatch.setattr(
        linear, "metrics", lambda predictions, targets: {"accuracy": next(scores)}
    )
    tuning.tune(c, trials=2, id="pruning", root=tmp_path)
    pruned = optuna.load_study(
        study_name="pruning",
        storage=tuning.storage(tmp_path / "experiments/studies/pruning"),
    )
    assert isinstance(pruners[-1], optuna.pruners.MedianPruner)
    assert [trial.state.name for trial in pruned.trials] == ["COMPLETE", "PRUNED"]
    assert list(pruned.trials[1].intermediate_values) == [1, 2]
    expected = {
        "strategy": "median",
        "params": {
            "n_startup_trials": 1,
            "n_warmup_steps": 2,
            "interval_steps": 1,
            "n_min_trials": 1,
        },
    }
    study_path = tmp_path / "experiments/studies/pruning"
    for file in [
        study_path / "config.toml",
        study_path / "config-1.toml",
        tmp_path
        / "experiments/runs"
        / pruned.trials[0].user_attrs["run_id"]
        / "config.toml",
    ]:
        assert load(file).search.pruning.model_dump() == expected
    assert pruned.trials[0].user_attrs["config"]["search"]["pruning"] == expected
    exported = tuning.export(c, "pruning", "median-selected.toml", root=tmp_path)
    assert load(exported).search.pruning.model_dump() == expected
    with pytest.raises(ValueError):
        tuning.tune(
            cfg(tmp_path), trials=1, id="pruning", continuation=True, root=tmp_path
        )
    unsupported = cfg(tmp_path, "ragged")
    unsupported.search.pruning = c.search.pruning
    monkeypatch.setattr(ragged, "supports_pruning", False)
    with pytest.raises(ValueError, match="does not support pruning"):
        tuning.tune(unsupported, trials=1, root=tmp_path)
    with pytest.raises(ValueError, match="does not support pruning"):
        runner.execute(unsupported, trial=object(), root=tmp_path)
    with pytest.raises(ValueError, match="params"):
        tuning.sample(
            c.model_copy(
                update={
                    "search": c.search.model_copy(
                        update={
                            "space": {
                                "data.params.size": {"type": "int", "low": 1, "high": 2}
                            }
                        }
                    )
                }
            ),
            optuna.trial.FixedTrial({}),
        )


def test_groups_queries_statistics_and_saved_plots(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    runner.prepare(c, tmp_path)
    c.provenance = {"study": "chosen-study", "trial": 2}
    group = groups.create([c], [1, 2, 3], id="seeds", root=tmp_path)
    original = linear.build
    calls = 0

    def fail_second(config, root):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TrialFailure("second member")
        return original(config, root)

    monkeypatch.setattr(linear, "build", fail_second)
    with pytest.raises(TrialFailure):
        groups.run_group(c, group, root=tmp_path)
    meta = analysis.detail(group, root=tmp_path, folder="groups")
    members = meta["members"]
    first = tmp_path / "experiments/runs" / members[0] / "record.json"
    before = first.read_bytes()
    with pytest.raises(ValueError):
        groups.run_group(c, group, root=tmp_path)
    with pytest.raises(ValueError):
        groups.run_group(c, group, action="resume", root=tmp_path)
    monkeypatch.setattr(linear, "build", original)
    groups.run_group(c, group, action="rerun", root=tmp_path)
    assert first.read_bytes() == before
    group_meta = analysis.detail(group, folder="groups", root=tmp_path)
    assert "error" not in group_meta
    assert (
        group_meta["name"] is None
        and group_meta["tags"] == []
        and group_meta["note"] == ""
    )
    assert read_json(first)["source"] == c.provenance | {"group": group}
    assert load(
        tmp_path / "experiments/groups" / group / f"{members[0]}.toml"
    ).provenance == c.provenance | {"group": group}
    analysis.annotate(
        group, folder="groups", name="named", note="purpose", root=tmp_path
    )
    assert analysis.detail(group, folder="groups", root=tmp_path)["name"] == "named"
    resumed_group = groups.create([c], [7, 8], id="resume-group", root=tmp_path)
    original_log = Run.log

    def interrupt_member(self, values, epoch, split="validation"):
        original_log(self, values, epoch, split)
        if self.config.seed == 8 and epoch == 2:
            raise TrialFailure("member has a complete checkpoint")

    monkeypatch.setattr(Run, "log", interrupt_member)
    with pytest.raises(TrialFailure):
        groups.run_group(c, resumed_group, root=tmp_path)
    monkeypatch.setattr(Run, "log", original_log)
    groups.run_group(c, resumed_group, action="resume", root=tmp_path)
    assert (
        analysis.detail(resumed_group, folder="groups", root=tmp_path)["status"]
        == "completed"
    )
    evaluations = groups.test_members(c, group, members, root=tmp_path)
    analysis.annotate(evaluations[0], root=tmp_path, tags=["chosen"], note="purpose")
    assert len(analysis.list_records(root=tmp_path, tag="chosen")) == 1
    assert analysis.summary(evaluations, "accuracy", root=tmp_path)["n"] == 3
    analysis.table(
        evaluations,
        "accuracy",
        destination="experiments/reports/table.csv",
        root=tmp_path,
    )
    fig = plotting.curves(members, "accuracy", split="validation", root=tmp_path)
    plotting.save_figure(fig, "experiments/reports/curve.png", root=tmp_path)
    plotting.save_figure(
        plotting.classification([0, 1], [0, 0]),
        "experiments/reports/confusion.pdf",
        root=tmp_path,
    )
    with pytest.raises(FileExistsError):
        analysis.table(
            evaluations,
            "accuracy",
            destination="experiments/reports/table.csv",
            root=tmp_path,
        )
    analysis.table(
        evaluations,
        "accuracy",
        destination="experiments/reports/table.csv",
        overwrite=True,
        root=tmp_path,
    )
    with pytest.raises(FileExistsError):
        plotting.save_figure(fig, "experiments/reports/curve.png", root=tmp_path)
    plotting.save_figure(
        fig, "experiments/reports/curve.png", dpi=100, overwrite=True, root=tmp_path
    )
    atomic(tmp_path / "input.toml", dumps(c))
    for args in [
        [
            "compare",
            *evaluations,
            "--metric",
            "accuracy",
            "--csv",
            "cli.csv",
            "--save",
            "comparison.json",
        ],
        [
            "plot",
            members[0],
            "--metric",
            "loss",
            "--split",
            "train",
            "--destination",
            "cli.png",
            "--dpi",
            "100",
        ],
        ["group", "create", "input.toml", "--seed", "9", "--id", "cli-group"],
    ]:
        result = CliRunner().invoke(app, args + ["--root", str(tmp_path)])
        assert result.exit_code == 0, result.output
    assert read_json(tmp_path / "comparison.json") == analysis.table(
        evaluations, "accuracy", root=tmp_path
    )
    # Deliberately controlled saved scores verify pairing/input checks, not model superiority.
    right = []
    for i, source in enumerate(members):
        source_config = load(tmp_path / "experiments/runs" / source / "config.toml")
        id = runner.test(source_config, source=source, root=tmp_path)
        right.append(id)
        for run_id, value in [
            (evaluations[i], [0.6, 0.7, 0.9][i]),
            (id, [0.5, 0.65, 0.7][i]),
        ]:
            file = tmp_path / "experiments/runs" / run_id / "record.json"
            record = read_json(file)
            record["result"]["accuracy"] = value
            atomic(file, record)
    result = analysis.paired(
        evaluations, right, "accuracy", test="ttest", root=tmp_path
    )
    assert result["pairs"] == 3 and 0 <= result["p_value"] <= 1
    assert result["left"] == pytest.approx(
        {"mean": np.mean([0.6, 0.7, 0.9]), "std": np.std([0.6, 0.7, 0.9], ddof=1)}
    )
    assert result["right"] == pytest.approx(
        {"mean": np.mean([0.5, 0.65, 0.7]), "std": np.std([0.5, 0.65, 0.7], ddof=1)}
    )
    assert result["mean_difference"] == pytest.approx((0.1 + 0.05 + 0.2) / 3)
    assert result["alternative"] == "two-sided" and result["status"] == "ok"
    assert "significant" not in result and "reason" not in result
    arguments = ["paired", "accuracy", "--test", "ttest", "--root", str(tmp_path)]
    for flag, ids in [("--left", evaluations), ("--right", right)]:
        for id in ids:
            arguments += [flag, id]
    cli_result = CliRunner().invoke(app, arguments + ["--save", "paired.json"])
    assert cli_result.exit_code == 0, cli_result.output
    assert "Paired comparison" in cli_result.output
    assert read_json(tmp_path / "paired.json") == result
    wilcoxon = analysis.paired(
        evaluations, right, "accuracy", test="wilcoxon", root=tmp_path
    )
    assert wilcoxon["pairs"] == 3 and wilcoxon["status"] == "ok"
    assert 0 <= wilcoxon["p_value"] <= 1 and wilcoxon["alternative"] == "two-sided"

    def unexpected(*args, **kwargs):
        raise RuntimeError("unexpected statistical library error")

    with monkeypatch.context() as patch:
        patch.setattr(analysis.stats, "ttest_rel", unexpected)
        with pytest.raises(RuntimeError, match="unexpected statistical"):
            analysis.paired(evaluations, right, "accuracy", test="ttest", root=tmp_path)
    for test, left_scores, right_scores, difference in [
        ("ttest", [2, 3, 4], [1, 2, 3], 1),
        ("ttest", [0.1, 0.1, 0.1], [0, 0, 0], 0.1),
        ("wilcoxon", [1, 2, 3], [1, 2, 3], 0),
    ]:
        for ids, scores in [(evaluations, left_scores), (right, right_scores)]:
            for id, score in zip(ids, scores, strict=True):
                file = tmp_path / "experiments/runs" / id / "record.json"
                record = read_json(file)
                record["result"]["accuracy"] = score
                atomic(file, record)
        degenerate = analysis.paired(
            evaluations, right, "accuracy", test=test, root=tmp_path
        )
        assert degenerate["p_value"] is None and degenerate["status"] == "not_computed"
        assert degenerate["reason"] and degenerate["alternative"] == "two-sided"
        assert degenerate["pairs"] == 3 and degenerate[
            "mean_difference"
        ] == pytest.approx(difference)
        assert degenerate["left"] == pytest.approx(
            {"mean": np.mean(left_scores), "std": np.std(left_scores, ddof=1)}
        )
        assert degenerate["right"] == pytest.approx(
            {"mean": np.mean(right_scores), "std": np.std(right_scores, ddof=1)}
        )
        cli_result = CliRunner().invoke(
            app, arguments + ["--test", test, "--save", "paired.json", "--overwrite"]
        )
        assert cli_result.exit_code == 0, cli_result.output
        assert "not_computed" in cli_result.output
        assert read_json(tmp_path / "paired.json") == degenerate
    for left_ids, right_ids in [
        (evaluations, evaluations),
        ([evaluations[0], evaluations[0]], right[:2]),
        (evaluations, right[:2]),
        (evaluations[:1], right[:1]),
    ]:
        with pytest.raises(ValueError):
            analysis.paired(
                left_ids, right_ids, "accuracy", test="ttest", root=tmp_path
            )
    file = tmp_path / "experiments/runs" / right[0] / "record.json"
    original = file.read_text(encoding="utf-8")
    record = read_json(file)
    record["result"]["accuracy"] = float("nan")
    file.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        analysis.paired(evaluations, right, "accuracy", test="ttest", root=tmp_path)
    file.write_text(original, encoding="utf-8")
    snapshot = file.parent / "config-1.toml"
    test_config = load(snapshot)
    test_config.protocol.split = "different"
    atomic(snapshot, dumps(test_config))
    assert analysis.table([evaluations[0], right[0]], "accuracy", root=tmp_path)[0][
        "warning"
    ]
    with pytest.raises(ValueError):
        analysis.paired(evaluations, right, "accuracy", test="ttest", root=tmp_path)


def test_tensorboard_and_platform_failure(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    c.platforms.tensorboard = True
    runner.prepare(c, tmp_path)
    runner.execute(c, id="tb", root=tmp_path)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = list((tmp_path / "experiments/runs/tb").rglob("events.out.tfevents.*"))
    assert files
    events = EventAccumulator(str(files[0])).Reload()
    assert [e.step for e in events.Scalars("validation/accuracy")] == list(
        range(1, c.train.epochs + 1)
    )
    assert [e.step for e in events.Scalars("train/loss")] == list(
        range(1, c.train.epochs + 1)
    )
    from core.platforms import Platforms

    def failure(self, values, step):
        raise PlatformError("enabled platform failed")

    monkeypatch.setattr(Platforms, "log", failure)
    with pytest.raises(PlatformError):
        runner.execute(c, id="platform-failure", root=tmp_path)
    assert (
        read_json(tmp_path / "experiments/runs/platform-failure/record.json")["status"]
        == "failed"
    )
