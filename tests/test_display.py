"""CLI presentation must not change research records or numeric results."""

from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import torch
from rich.console import Console
from typer.testing import CliRunner

from core import analysis, display, runner
from core.cli import app
from core.config import dumps
from core.records import checkpoint, history, read_json
from tests.test_workflow import cfg


def test_query_views_and_lossless_exports(tmp_path, monkeypatch):
    score = 0.912345678901234
    row = {
        "id": "a" * 32,
        "kind": "train",
        "method": "linear",
        "dataset": "A",
        "seed": 1,
        "status": "completed",
        "name": "[red]literal[/red]",
        "result": {"accuracy": score},
        "launches": [
            {
                "number": 1,
                "status": "failed",
                "error": "historical failure",
                "environment": {"hidden": "preserved"},
            }
        ],
    }
    original = deepcopy(row)
    monkeypatch.setattr(analysis, "list_records", lambda *args, **kwargs: [row])
    monkeypatch.setattr(analysis, "detail", lambda *args, **kwargs: row)
    cli = CliRunner()
    for args in (["list"], ["show", row["id"]]):
        result = cli.invoke(
            app,
            args + ["--root", str(tmp_path), "--save", "result.json", "--overwrite"],
        )
        assert result.exit_code == 0, result.output
        assert "Saved JSON:" in result.output
        saved = read_json(tmp_path / "result.json")
        assert saved == ([row] if args[0] == "list" else row)
        if args[0] == "show":
            assert "0.912346" in result.output and "historical failure" in result.output
            assert "preserved" not in result.output
    assert row == original
    previous = (tmp_path / "result.json").read_bytes()
    result = cli.invoke(app, ["list", "--root", str(tmp_path), "--save", "result.json"])
    assert isinstance(result.exception, FileExistsError)
    assert "Saved JSON:" not in result.output
    assert (tmp_path / "result.json").read_bytes() == previous
    assert cli.invoke(app, ["list", "--json"]).exit_code == 2
    monkeypatch.setattr(analysis, "list_records", lambda *args, **kwargs: [])
    result = cli.invoke(app, ["list", "--root", str(tmp_path), "--save", "empty.json"])
    assert "No matching records." in result.output
    assert read_json(tmp_path / "empty.json") == []
    summary = {"n": 3, "mean": score, "std": 0.00000000321789}
    monkeypatch.setattr(analysis, "summary", lambda *args, **kwargs: summary)
    result = cli.invoke(
        app,
        [
            "summarize",
            "a",
            "b",
            "--metric",
            "score",
            "--root",
            str(tmp_path),
            "--save",
            "summary.json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3.21789e-09" in result.output
    assert read_json(tmp_path / "summary.json") == summary
    stream = StringIO()
    monkeypatch.setattr(display, "console", Console(file=stream, width=180))
    rows = [
        {
            "id": "a",
            "method": "linear",
            "dataset": "A",
            "seed": 1,
            "value": score,
            "setup": {"params": {"lr": 0.1}},
            "protocol": {"split": "one"},
            "warning": "INCOMPATIBLE",
        },
        {
            "id": "b",
            "method": "linear",
            "dataset": "B",
            "seed": 2,
            "value": score,
            "setup": {"params": {"lr": 0.2}},
            "protocol": {"split": "two"},
            "warning": "INCOMPATIBLE",
        },
    ]
    before = deepcopy(rows)
    display.comparison(rows, "score")
    display.detail(
        row
        | {
            "kind": "study",
            "trials": [
                {
                    "number": 0,
                    "status": "FAIL",
                    "value": None,
                    "params": {},
                    "run_id": None,
                    "retry_of": None,
                    "error": "trial error",
                }
            ],
        }
    )
    display.detail(
        {
            "id": "group",
            "kind": "group",
            "status": "completed",
            "member_states": {"member": "completed"},
        }
    )
    assert all(
        text in stream.getvalue()
        for text in (
            "INCOMPATIBLE",
            "Training differences",
            "Protocol differences",
            "trial error",
            "Members",
        )
    )
    assert rows == before


def test_training_feedback_preserves_records(tmp_path, monkeypatch):
    config = cfg(tmp_path)
    runner.prepare(config, tmp_path)
    runner.execute(config, id="silent", root=tmp_path)
    stream = StringIO()
    monkeypatch.setattr(display, "console", Console(file=stream, width=140))
    runner.execute(config, id="visible", root=tmp_path, display=display.RunDisplay())
    silent, visible = [
        tmp_path / "experiments/runs" / id for id in ("silent", "visible")
    ]
    assert history(silent) == history(visible)
    assert len(history(visible)) == 2 * config.train.epochs
    assert (
        read_json(silent / "record.json")["result"]
        == read_json(visible / "record.json")["result"]
    )
    a, b = [
        torch.load(checkpoint(directory, "last"), weights_only=True)
        for directory in (silent, visible)
    ]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    assert len(list(silent.iterdir())) == len(list(visible.iterdir()))
    text = stream.getvalue()
    assert "Starting train · visible" in text and "Completed train · visible" in text
    assert text.count("train/loss=") == config.train.epochs
    assert "\x1b[" not in text and "\r" not in text
    stream.seek(0)
    stream.truncate(0)
    runner.test(
        config,
        source="visible",
        id="evaluation",
        root=tmp_path,
        display=display.RunDisplay(),
    )
    text = stream.getvalue()
    assert "Starting test · evaluation" in text and "Checkpoint" not in text
    assert "Epoch " not in text and "Completed test · evaluation" in text
    before = (visible / "record.json").read_bytes()
    runner.execute(
        config, id="visible", resume=True, root=tmp_path, display=display.RunDisplay()
    )
    assert (visible / "record.json").read_bytes() == before
    # Exercise Rich's terminal branch without a snapshot test or real terminal.
    terminal = StringIO()
    monkeypatch.setattr(
        display, "console", Console(file=terminal, force_terminal=True, width=140)
    )
    runner.execute(config, id="terminal", root=tmp_path, display=display.RunDisplay())
    assert "Epoch " in terminal.getvalue() and "\x1b[" in terminal.getvalue()


def test_custom_failure_and_optuna_output_boundary(tmp_path, monkeypatch):
    config = cfg(tmp_path)
    stream = StringIO()
    monkeypatch.setattr(display, "console", Console(file=stream, width=140))

    def custom(config, run, root, trial):
        run.log({"score": 0.123456789}, step=20)
        return {"score": 0.987654321}

    monkeypatch.setattr(runner, "get", lambda name: SimpleNamespace(run=custom))
    runner.execute(config, id="custom", root=tmp_path, display=display.RunDisplay())
    assert "Epoch " not in stream.getvalue()
    assert "0.987654" in stream.getvalue()
    assert (
        read_json(tmp_path / "experiments/runs/custom/record.json")["result"]["score"]
        == 0.987654321
    )
    stream.seek(0)
    stream.truncate(0)
    runner.execute(
        config, id="trial", root=tmp_path, trial=object(), display=display.RunDisplay()
    )
    assert stream.getvalue() == ""

    def failed(config, run, root, trial):
        raise ValueError("[red]original error[/red]")

    monkeypatch.setattr(runner, "get", lambda name: SimpleNamespace(run=failed))
    (tmp_path / "input.toml").write_text(dumps(config), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["train", "input.toml", "--id", "failed", "--root", str(tmp_path)]
    )
    assert isinstance(result.exception, ValueError)
    text = stream.getvalue()
    assert "Failed train · failed" in text and "[red]original error[/red]" in text
    assert "Results" not in text
    assert (
        read_json(tmp_path / "experiments/runs/failed/record.json")["status"]
        == "failed"
    )


def test_compact_details_history_and_utc_times(monkeypatch, tmp_path):
    stream = StringIO()
    monkeypatch.setattr(display, "console", Console(file=stream, width=240))
    start = "2026-09-18T18:00:00.123456+08:00"
    end = "2026-09-18T18:04:32.654321+08:00"
    record = {
        "id": "failed-run",
        "kind": "train",
        "status": "failed",
        "created": start,
        "ended": end,
        "error": "original failure",
        "protocol": {"task": "classification"},
        "launches": [
            {
                "number": 1,
                "status": "failed",
                "started": start,
                "ended": end,
                "resume": None,
                "error": "original failure",
            }
        ],
    }
    original = deepcopy(record)
    display.detail(record)
    text = stream.getvalue()
    assert "Launch history" not in text
    assert text.count("original failure") == 1
    assert "2026-09-18 10:00:00" in text and "2026-09-18 10:04:32" in text
    assert "Started (UTC)" in text and "Ended (UTC)" in text
    assert "10:00:00 UTC" not in text and "10:04:32 UTC" not in text
    assert "123456" not in text and "+08:00" not in text
    assert "Protocol.task" in text
    assert record == original
    analysis.save_json(record, "record.json", root=tmp_path)
    assert read_json(tmp_path / "record.json") == original
    record["launches"].append(
        {
            "number": 2,
            "status": "failed",
            "started": end,
            "ended": end,
            "resume": {"launch": 1, "epoch": 2},
            "error": "latest failure",
        }
    )
    record["error"] = "latest failure"
    stream.seek(0)
    stream.truncate(0)
    display.detail(record)
    text = stream.getvalue()
    assert "Launch history" in text and "Launch errors" in text
    assert text.count("original failure") == text.count("latest failure") == 1
    record["launches"][0]["error"] = "latest failure"
    stream.seek(0)
    stream.truncate(0)
    display.detail(record)
    assert stream.getvalue().count("latest failure") == 2
    record["launches"][0]["error"] = "original failure"
    record.update(status="completed", error=None)
    record["launches"][-1].update(status="completed", error=None)
    stream.seek(0)
    stream.truncate(0)
    display.detail(record)
    assert stream.getvalue().count("original failure") == 1
    assert "latest failure" not in stream.getvalue()
    rendered = []
    monkeypatch.setattr(
        display, "fields", lambda title, data: rendered.append((title, data))
    )
    monkeypatch.setattr(
        display,
        "table",
        lambda title, columns, rows: rendered.append((title, list(rows))),
    )
    paired = {
        "test": "ttest",
        "alternative": "two-sided",
        "pairs": 3,
        "mean_difference": 1.0,
        "p_value": None,
        "status": "not_computed",
        "reason": "Paired differences have zero variance",
        "left": {"mean": 2.0, "std": 0.0},
        "right": {"mean": 1.0, "std": 0.0},
    }
    display.paired(paired, "score")
    assert len(rendered) == 2
    assert rendered[0][1][0]["Test"] == "ttest"
    assert rendered[0][1][1]["Reason"] == paired["reason"]


def test_grouped_fields_and_compact_listing(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(display, "console", Console(file=stream, width=100))
    records = [
        dict(
            id="274ca1aa436845fc9bfb8393403f9e65",
            name="Baseline classification",
            kind="train",
            method="linear",
            dataset="toy-classification",
            seed=42,
            status="completed",
        ),
        dict(
            id="75f6899000574a3d96ecec0952ea5f6e",
            name="",
            kind="train",
            method="linear",
            dataset="toy-classification",
            seed=7,
            status="completed",
        ),
    ]
    original = deepcopy(records)
    display.listing(records)
    text = stream.getvalue()
    assert "Experiment" in text and "Baseline classification" in text
    assert all(text.count(record["id"]) == 1 for record in records)
    assert records == original
    views = []
    monkeypatch.setattr(display.console, "print", views.append)
    display.fields("Sections", [{"One": 1}, {}, {"Two": 2}, {}])
    assert [row.end_section for row in views[-1].rows] == [True, False]
    display.detail(
        dict(
            id="study",
            kind="study",
            status="completed",
            trials=[],
            trial_counts={"COMPLETE": 1, "FAIL": 0, "PRUNED": 0},
            unfinished_trials=[],
        )
    )
    labels = views[-2].columns[0]._cells
    assert {str(cell) for cell in labels} >= {
        "COMPLETE",
        "FAIL",
        "PRUNED",
        "Unfinished trials",
    }
