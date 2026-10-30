"""Optuna owns trials and scheduling; each trial is an ordinary independent run."""

from importlib import import_module
from math import prod

import optuna

from core import storage as artifacts
from core.config import ROOT, Config, changed, dumps, load, set_value
from core.platforms import PlatformError
from core.records import atomic, directory, fresh, identifier, locked, now, read_json
from core.runner import execute
from methods import get


def storage(target):
    return optuna.storages.RDBStorage(
        f"sqlite:///{(target / 'study.sqlite3').as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


def exceptions(names):
    result = []
    for name in names:
        module, attribute = name.rsplit(".", 1)
        cls = getattr(import_module(module), attribute)
        if (
            not isinstance(cls, type)
            or not issubclass(cls, Exception)
            or cls in (Exception, RuntimeError)
            or issubclass(PlatformError, cls)
            or issubclass(cls, PlatformError)
        ):
            raise ValueError(
                "catch must name specific exception types, never Exception/RuntimeError/platform errors"
            )
        result.append(cls)
    return tuple(result)


def sample(config, trial):
    data = config.model_dump()
    for name, specification in config.search.space.items():
        if not name.startswith("params."):
            raise ValueError(
                "Configured search parameters must be under params.*; use method.suggest for complex search"
            )
        spec = dict(specification)
        kind = spec.pop("type")
        if kind not in ("float", "int", "categorical"):
            raise ValueError(f"Unknown search distribution: {kind}")
        value = getattr(trial, f"suggest_{kind}")(name, **spec)
        set_value(data, name, value)
    method = get(config.method)
    if hasattr(method, "suggest"):
        data = method.suggest(trial, data)
    actual = Config.model_validate(data)
    changed(config, actual, {"params"})
    return actual


def tune(config, *, trials, id=None, continuation=False, retry=None, root=ROOT):
    if (
        trials < 1
        or (continuation and id is None)
        or (retry is not None and (not continuation or retry < 0))
    ):
        raise ValueError(
            "Positive additional trial budget required; retry requires explicit continuation"
        )
    # Notebook callers may have edited a Config in memory; resolve before recording.
    config = Config.model_validate(config.model_dump())
    method = get(config.method)
    config.train.require("direction")
    if not hasattr(method, "objective"):
        config.train.require("metric")
    choice = config.search.sampler
    if choice.strategy == "grid":
        grid = {}
        for name, spec in config.search.space.items():
            if not name.startswith("params.") or spec.get("type") != "categorical":
                raise ValueError(
                    "Grid requires predefined params.* categorical choices"
                )
            if spec.keys() != {"type", "choices"} or not spec["choices"]:
                raise ValueError("Grid requires nonempty categorical choices only")
            grid[name] = spec["choices"]
        sampler = optuna.samplers.GridSampler(grid, seed=config.seed)
        if not continuation:
            print(f"Grid combinations: {prod(len(values) for values in grid.values())}")
    else:
        sampler = optuna.samplers.TPESampler(seed=config.seed, **choice.params)
    caught = exceptions(config.search.catch)
    if config.search.pruning.strategy != "none" and not getattr(
        get(config.method), "supports_pruning", False
    ):
        raise ValueError("This method does not support pruning")
    id = identifier(id)
    target = directory(config, "studies", id, root)
    # SQLite/Optuna handles concurrent trials. Only metadata initialization/updates lock.
    with locked(target):
        if continuation:
            changed(
                load(target / "config.toml", root=root),
                config,
                {"device", "platforms", "search.catch", "storage"},
            )
            meta = read_json(target / "record.json")
        else:
            fresh(target)
            atomic(target / "config.toml", dumps(config))
            meta = {
                "id": id,
                "kind": "study",
                "method": config.method,
                "dataset": config.data.name,
                "name": config.name,
                "tags": config.tags,
                "note": config.note,
                "created": now(),
                "launches": [],
            }
        study = optuna.create_study(
            study_name=id,
            storage=storage(target),
            direction={"min": "minimize", "max": "maximize"}[config.train.direction],
            load_if_exists=continuation,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(**config.search.pruning.params)
            if config.search.pruning.strategy == "median"
            else optuna.pruners.NopPruner(),
        )
        if retry is not None:
            failed = study.trials[retry]
            if failed.number != retry or failed.state != optuna.trial.TrialState.FAIL:
                raise ValueError("Only a failed trial can be retried")
            study.enqueue_trial(failed.params, user_attrs={"retry_of": retry})

        launch = len(meta["launches"]) + 1
        meta["launches"].append(
            {"number": launch, "started": now(), "status": "running", "trials": trials}
        )
        meta["status"] = "running"
        atomic(target / f"config-{launch}.toml", dumps(config))
        atomic(target / "record.json", meta)

    def objective(trial):
        run_id = identifier()
        trial.set_user_attr("run_id", run_id)
        try:
            actual = sample(config, trial)
            trial.set_user_attr("config", actual.model_dump(exclude_none=True))
            execute(
                actual,
                id=run_id,
                root=root,
                trial=trial,
                source={
                    "study": id,
                    "trial": trial.number,
                    "retry_of": trial.user_attrs.get("retry_of"),
                },
                _upload=False,
            )
            method = get(actual.method)
            if hasattr(method, "objective"):
                from core.records import numbers

                return numbers(
                    {
                        "objective": method.objective(
                            actual, directory(actual, "runs", run_id, root), trial
                        )
                    }
                )["objective"]
            # Default objective is best validation only, never any test result.
            from core.records import history

            values = [
                row["metrics"][config.train.metric]
                for row in history(directory(config, "runs", run_id, root))
                if row["split"] == "validation"
            ]
            if not values:
                raise ValueError("Custom search objective must log validation metrics")
            return (max if config.train.direction == "max" else min)(values)
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

    status, error = "completed", None
    try:
        if not (
            isinstance(sampler, optuna.samplers.GridSampler)
            and sampler.is_exhausted(study)
            and retry is None
        ):
            study.optimize(
                objective, n_trials=trials, catch=caught, gc_after_trial=True
            )
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with locked(target):
            meta = read_json(target / "record.json")
            meta["launches"][launch - 1].update(status=status, error=error, ended=now())
            meta["status"] = (
                "running"
                if any(item["status"] == "running" for item in meta["launches"])
                else status
            )
            atomic(target / "record.json", meta)
    artifacts.auto_upload(config, "studies", id, overwrite=continuation, root=root)
    return id


def export(
    config, study_id, destination, *, trial_number=None, overwrite=False, root=ROOT
):
    from core.config import path

    target = directory(config, "studies", study_id, root)
    study = optuna.load_study(study_name=study_id, storage=storage(target))
    if trial_number is not None and trial_number < 0:
        raise ValueError("Trial number must be nonnegative")
    trial = study.best_trial if trial_number is None else study.trials[trial_number]
    if trial.state != optuna.trial.TrialState.COMPLETE:
        raise ValueError("Export requires a completed trial")
    actual = Config.model_validate(
        trial.user_attrs["config"]
        | {"provenance": {"study": study_id, "trial": trial.number}}
    )
    output = path(destination, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8") as file:
        file.write(f"# source study={study_id} trial={trial.number}\n" + dumps(actual))
    return output
