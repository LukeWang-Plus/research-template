"""Focused checks for the confirmed sampler and logging choices."""

from inspect import signature
from types import SimpleNamespace

import optuna
import pytest
from pydantic import ValidationError

from core import analysis, runner, tuning
from core.config import Sampler, load
from core.platforms import Platforms
from core.records import Run, atomic, history, locked, read_json
from tests.test_workflow import cfg


def test_sampler_grid_and_study_status(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    names = (
        "n_startup_trials",
        "n_ei_candidates",
        "multivariate",
        "group",
        "constant_liar",
    )
    native = signature(optuna.samplers.TPESampler)
    expected = {
        name: native.parameters[name].default
        for name in names
        if native.parameters[name].default is not None
    }
    assert c.search.sampler.params == expected
    # Simulate a dependency default change without running or replacing the sampler.
    with monkeypatch.context() as patch:
        patch.setattr(
            optuna.samplers.TPESampler,
            "__signature__",
            native.replace(
                parameters=[
                    p.replace(default=17) if p.name == "n_startup_trials" else p
                    for p in native.parameters.values()
                ]
            ),
            raising=False,
        )
        assert Sampler().params["n_startup_trials"] == 17
        assert Sampler(
            params={"n_startup_trials": 3, "multivariate": False}
        ).params == expected | {"n_startup_trials": 3, "multivariate": False}
    for params in (
        {"group": True, "multivariate": False},
        {"n_ei_candidates": 0},
        {"unknown": 1},
        {"n_startup_trials": "2"},
    ):
        with pytest.raises(ValidationError):
            Sampler(params=params)
    # Cheap custom objective checks sampling, not training quality.
    method = SimpleNamespace(
        run=lambda config, run, root, trial: {"value": config.params["lr"]},
        objective=lambda config, directory, trial: config.params["lr"],
    )
    monkeypatch.setattr(runner, "get", lambda name: method)
    monkeypatch.setattr(tuning, "get", lambda name: method)
    c.train.metric = None
    c.train.epochs = None
    for id in ("tpe-a", "tpe-b"):
        tuning.tune(c, trials=2, id=id, root=tmp_path)
    a, b = [
        analysis.detail(id, folder="studies", root=tmp_path)
        for id in ("tpe-a", "tpe-b")
    ]
    assert [t["params"] for t in a["trials"]] == [t["params"] for t in b["trials"]]
    for trial in a["trials"]:
        actual = load(tmp_path / "experiments/runs" / trial["run_id"] / "config.toml")
        assert actual.seed == c.seed and actual.search.sampler.params == expected
    c.search.sampler = Sampler(strategy="grid")
    with pytest.raises(ValueError, match="categorical"):
        tuning.tune(c, trials=1, root=tmp_path)
    c.search.space = {"params.lr": {"type": "categorical", "choices": [0.01, 0.1]}}
    tuning.tune(c, trials=5, id="grid", root=tmp_path)
    tuning.tune(c, trials=2, id="grid", continuation=True, root=tmp_path)
    target = tmp_path / "experiments/studies/grid"
    detail = analysis.detail("grid", folder="studies", root=tmp_path)
    assert len(detail["trials"]) == 2
    assert {t["params"]["params.lr"] for t in detail["trials"]} == {0.01, 0.1}
    meta = read_json(target / "record.json")
    meta["launches"][0]["status"] = "running"
    atomic(target / "record.json", meta)
    before = (target / "record.json").read_bytes()
    snapshot = analysis.detail("grid", folder="studies", root=tmp_path)
    assert snapshot["status"] == "unconfirmed"
    assert snapshot["trial_counts"] == {"COMPLETE": 2, "FAIL": 0, "PRUNED": 0}
    assert snapshot["unfinished_trials"] == []
    assert (target / "record.json").read_bytes() == before
    method.suggest = lambda trial, data: (
        data | {"params": {"lr": trial.suggest_float("unlisted", 0.0, 1.0)}}
    )
    with pytest.raises(ValueError, match="unlisted"):
        tuning.tune(c, trials=1, id="bad-grid", root=tmp_path)


def test_platform_epoch_batching_and_custom_coordinate(tmp_path):
    c = cfg(tmp_path)
    platforms = Platforms(c, tmp_path, "buffer-check")
    emitted = []
    platforms.writers.append(lambda values, step: emitted.append((step, values)))
    platforms.log({"train/loss": 1.0}, 1)
    platforms.log({"validation/loss": 2.0}, 1)
    platforms.log({"train/loss": 0.5}, 2)
    platforms.log({"validation/loss": 1.5}, 2)
    platforms.close()
    assert emitted == [
        (1, {"train/loss": 1.0, "validation/loss": 2.0}),
        (2, {"train/loss": 0.5, "validation/loss": 1.5}),
    ]
    target = tmp_path / "custom"
    target.mkdir()
    with locked(target):
        run = Run(c, target, root=tmp_path)
        run.log({"reward": 2}, step=10, split="train")
        run.save_checkpoint({"updates": 10}, step=10)
        run.finish(
            "completed",
            result={"reward": 2},
            result_source={"kind": "custom_final", "launch": 1},
        )
    assert history(target)[0]["step"] == 10
    assert "epoch" not in read_json(target / "checkpoint.json")


def test_platform_finish_status_and_sdk_errors(tmp_path, monkeypatch):
    import sys

    import optuna

    from core.platforms import PlatformError

    c = cfg(tmp_path)
    c.platforms.wandb = {"project": "local-check"}
    c.platforms.swanlab = {"project": "local-check"}
    for status, exception in (
        ("completed", None),
        ("failed", ValueError("research failed")),
        ("pruned", optuna.TrialPruned("pruned by study")),
    ):
        summary, finished, logged = {}, {}, []
        wandb_run = SimpleNamespace(
            summary=summary,
            log=lambda values, step: logged.append((values, step)),
            finish=lambda **kwargs: finished.update(wandb=kwargs),
        )
        monkeypatch.setitem(
            sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: wandb_run)
        )
        monkeypatch.setitem(
            sys.modules,
            "swanlab",
            SimpleNamespace(
                init=lambda **kwargs: None,
                log=lambda values, step: logged.append((values, step)),
                finish=lambda **kwargs: finished.update(swanlab=kwargs),
            ),
        )

        def custom(config, run, root, trial):
            run.log({"loss": 1.0}, 1, "train")
            if exception:
                raise exception
            return {"loss": 1.0}

        monkeypatch.setattr(runner, "get", lambda name: SimpleNamespace(run=custom))
        if exception:
            with pytest.raises(type(exception)):
                runner.execute(c, id=status, root=tmp_path)
        else:
            runner.execute(c, id=status, root=tmp_path)
        meta = read_json(tmp_path / "experiments/runs" / status / "record.json")
        assert meta["status"] == summary["research_status"] == status
        assert finished["wandb"] == {"exit_code": int(status == "failed")}
        assert finished["swanlab"] == {
            "state": "crashed" if status == "failed" else "success",
            "error": meta["error"],
        }
        if exception:
            assert summary["research_error"] == meta["error"]
        assert logged == [({"train/loss": 1.0}, 1)] * 2

    def fail_log(values, step):
        raise OSError("SDK exposed failure")

    wandb_run.log = fail_log
    platforms = Platforms(c, tmp_path, "sdk-failure")
    platforms.log({"loss": 1}, 1)
    with pytest.raises(PlatformError, match="SDK exposed failure"):
        platforms.close()
    assert finished["wandb"]["exit_code"] == 1
    assert finished["swanlab"]["state"] == "crashed"
