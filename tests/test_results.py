"""Regressions for result identity/configuration and checkpoint publication."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from core import analysis, plotting, records, runner
from core.config import Train, load
from core.records import checkpoint, checkpoint_info, history, read_json
from methods import linear
from tests.test_workflow import cfg


class SpecialState:
    def __init__(self, value):
        self.value = value


def special_method():
    def run(config, run, root, trial=None):
        start = 0
        if run.last:
            start = run.last["epoch"]
            assert run.last["state"]["special"].value == start
        for epoch in range(start + 1, config.train.epochs + 1):
            run.log({"accuracy": 100.0 - epoch}, epoch)
            run.save_checkpoint({"special": SpecialState(epoch)}, epoch, 100.0 - epoch)
        # Deliberately different from validation: custom comparisons use this result.
        return {"accuracy": -123.0}

    def test(config, run, root, file, state):
        return {"accuracy": state["special"].value}

    def load_state(file):
        # This test method explicitly trusts the objects it created.
        return torch.load(file, map_location="cpu", weights_only=False)

    return SimpleNamespace(
        run=run, test=test, load_state=load_state, supports_resume=True
    )


def test_final_only_and_dataset_identity(tmp_path, monkeypatch):
    method = SimpleNamespace(
        run=lambda config, run, root, trial: {"score": float(config.seed)}
    )
    monkeypatch.setattr(runner, "get", lambda name: method)
    configs = {}
    for id, seed, dataset in [
        ("a1", 1, "A"),
        ("a2", 2, "A"),
        ("b1", 1, "B"),
        ("b2", 2, "B"),
    ]:
        config = cfg(tmp_path, seed=seed)
        config.data.name = dataset
        config.data.path = None  # No fake path or epoch settings for in-memory methods.
        config.train = Train()
        runner.execute(config, id=id, root=tmp_path)
        configs[id] = config
    target = tmp_path / "experiments/runs/a1"
    assert not list(target.glob("metrics-*.jsonl"))
    assert not (target / "checkpoint.json").exists()
    assert (
        analysis.rows(["a1"], "score", root=tmp_path)[0]["result_source"]["kind"]
        == "custom_final"
    )
    assert analysis.summary(["a1", "a2"], "score", root=tmp_path)["mean"] == 1.5
    assert all(
        row["warning"] for row in analysis.table(["a1", "b1"], "score", root=tmp_path)
    )
    with pytest.raises(ValueError, match="Incompatible dataset"):
        analysis.summary(["a1", "b2"], "score", root=tmp_path)
    with pytest.raises(ValueError, match="Incompatible dataset"):
        analysis.paired(
            ["a1", "a2"], ["b1", "b2"], "score", test="ttest", root=tmp_path
        )
    configs["a2"].protocol.split = "another-split"
    runner.execute(configs["a2"], id="different-protocol", root=tmp_path)
    with pytest.raises(ValueError, match="Incompatible dataset"):
        analysis.summary(["a1", "different-protocol"], "score", root=tmp_path)
    with pytest.raises(ValueError, match="No committed"):
        plotting.curves(["a1"], "score", split="validation", root=tmp_path)


def test_completed_and_checkpoint_launch_configurations(tmp_path, monkeypatch):
    config = cfg(tmp_path)
    config.train.keep_top_k = 2
    runner.prepare(config, tmp_path)
    original_metrics = linear.metrics
    scores = iter([100.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(linear, "metrics", lambda p, t: {"accuracy": next(scores)})
    runner.execute(config, id="extended", root=tmp_path)
    extended = config.model_copy(
        update={"train": config.train.model_copy(update={"epochs": 10})}
    )
    runner.execute(extended, id="extended", resume=True, root=tmp_path)
    monkeypatch.setattr(linear, "metrics", original_metrics)
    row = analysis.rows(["extended"], "accuracy", root=tmp_path)[0]
    assert row["setup"]["train"]["epochs"] == 10
    assert row["value"] == 100.0
    assert row["result_source"] == {
        "kind": "validation_best",
        "launch": 2,
        "metric_launch": 1,
        "epoch": 1,
    }
    target = tmp_path / "experiments/runs/extended"
    assert load(target / "config.toml").train.epochs == 4
    second = config.model_copy(update={"seed": 2})
    runner.execute(second, id="four-epochs", root=tmp_path)
    with pytest.raises(ValueError, match="one method/setup"):
        analysis.summary(["extended", "four-epochs"], "accuracy", root=tmp_path)
    assert (
        analysis.table(["extended", "four-epochs"], "accuracy", root=tmp_path)[0][
            "setup"
        ]["train"]["epochs"]
        == 10
    )
    entry = next(
        item
        for item in read_json(target / "checkpoint.json")["top"]
        if item["launch"] == 2
    )
    evaluation = config.model_copy(deep=True)
    evaluation.data.name = "evaluation-dataset"
    evaluation.protocol.evaluation = "evaluation-settings"
    runner.test(evaluation, source="extended", id="old-best", root=tmp_path)
    runner.test(
        evaluation,
        source="extended",
        selection=entry["file"],
        id="new-candidate",
        root=tmp_path,
    )
    old, new = analysis.rows(["old-best", "new-candidate"], "accuracy", root=tmp_path)
    assert (old["setup"]["train"]["epochs"], new["setup"]["train"]["epochs"]) == (4, 10)
    assert new["dataset"] == "evaluation-dataset"
    assert new["protocol"]["evaluation"] == "evaluation-settings"
    assert (
        read_json(tmp_path / "experiments/runs/new-candidate/record.json")["source"][
            "launch"
        ]
        == 2
    )
    # External state has only the information explicitly supplied, not guessed provenance.
    runner.test(
        evaluation, external=target / entry["file"], id="external", root=tmp_path
    )
    assert (
        analysis.rows(["external"], "accuracy", root=tmp_path)[0]["setup"]["train"][
            "epochs"
        ]
        == 4
    )
    assert (
        read_json(tmp_path / "experiments/runs/external/record.json")["source"]["run"]
        is None
    )


def test_explicit_special_loader_and_metadata_only_queries(tmp_path, monkeypatch):
    method = special_method()
    monkeypatch.setattr(runner, "get", lambda name: method)
    config = cfg(tmp_path)
    config.train.epochs = 1
    runner.execute(config, id="special", root=tmp_path)
    target = tmp_path / "experiments/runs/special"
    trusted_loader = method.load_state
    del method.load_state
    config.train.epochs = 2
    with pytest.raises(pickle.UnpicklingError):
        runner.execute(config, id="special", resume=True, root=tmp_path)
    with pytest.raises(pickle.UnpicklingError):
        runner.test(config, source="special", root=tmp_path)
    method.load_state = trusted_loader
    runner.execute(config, id="special", resume=True, root=tmp_path)
    runner.test(config, source="special", id="special-test", root=tmp_path)
    runner.test(
        config, external=checkpoint(target), id="special-external", root=tmp_path
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Queries must never deserialize research state")

    monkeypatch.setattr(torch, "load", forbidden)
    assert checkpoint_info(target)["launch"] == 1
    assert [r["epoch"] for r in history(target)] == [1, 2]
    assert analysis.list_records(root=tmp_path)
    assert analysis.detail("special", root=tmp_path)["status"] == "completed"
    assert analysis.rows(["special"], "accuracy", root=tmp_path)[0]["value"] == -123.0
    assert analysis.rows(["special-test"], "accuracy", root=tmp_path)[0]["value"] == 1
    assert analysis.table(
        ["special-test", "special-external"], "accuracy", root=tmp_path
    )
    fig = plotting.curves(["special"], "accuracy", split="validation", root=tmp_path)
    assert list(fig.axes[0].lines[0].get_ydata()) == [99.0, 98.0]


@pytest.mark.parametrize("when", ["before", "after"])
def test_checkpoint_publication_interruption(tmp_path, monkeypatch, when):
    method = special_method()
    monkeypatch.setattr(runner, "get", lambda name: method)
    config = cfg(tmp_path)
    config.train.epochs = 1
    runner.execute(config, id="interrupted", root=tmp_path)
    target = tmp_path / "experiments/runs/interrupted"
    old = read_json(target / "checkpoint.json")
    old_bytes = (target / old["file"]).read_bytes()
    atomic = records.atomic

    def interrupt(file, value, **kwargs):
        if file.name == "checkpoint.json":
            assert (target / value["file"]).is_file()
            if when == "after":
                atomic(file, value, **kwargs)
            raise OSError("checkpoint publication interrupted")
        atomic(file, value, **kwargs)

    monkeypatch.setattr(records, "atomic", interrupt)
    config.train.epochs = 3
    with pytest.raises(OSError, match="publication interrupted"):
        runner.execute(config, id="interrupted", resume=True, root=tmp_path)
    committed = read_json(target / "checkpoint.json")
    assert committed["epoch"] == (1 if when == "before" else 2)
    assert (target / old["file"]).read_bytes() == old_bytes
    assert (
        method.load_state(checkpoint(target, "last"))["special"].value
        == committed["epoch"]
    )
    assert [r["epoch"] for r in history(target)] == list(
        range(1, committed["epoch"] + 1)
    )
    monkeypatch.setattr(records, "atomic", atomic)
    runner.execute(config, id="interrupted", resume=True, root=tmp_path)
    assert [(r["epoch"], r["launch"]) for r in history(target)] == [
        (1, 1),
        (2, 3 if when == "before" else 2),
        (3, 3),
    ]
    assert method.load_state(checkpoint(target, "last"))["special"].value == 3
    # Even repeated saves of the same epoch in one launch must use new filenames.
    with records.locked(target):
        run = records.Run(config, target, resume=True, root=tmp_path)
        run.save_checkpoint({"special": SpecialState(3)}, 3)
        first = read_json(target / "checkpoint.json")
        monkeypatch.setattr(records, "atomic", interrupt)
        with pytest.raises(OSError):
            run.save_checkpoint({"special": SpecialState(999)}, 3)
        assert (target / first["file"]).exists()
        if when == "before":
            assert read_json(target / "checkpoint.json") == first
            assert method.load_state(checkpoint(target, "last"))["special"].value == 3
        else:
            assert read_json(target / "checkpoint.json")["file"] != first["file"]
            assert method.load_state(checkpoint(target, "last"))["special"].value == 999
