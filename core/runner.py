"""Synchronous shared entry points for CLI and notebooks."""

import traceback
from pathlib import Path

import torch

from core import storage
from core.config import ROOT, Config, changed, load, path
from core.records import (
    Run,
    checkpoint_info,
    directory,
    fresh,
    history,
    identifier,
    load_state,
    locked,
    seed_all,
)
from core.training import evaluate, train
from methods import get


def prepare(config: Config, root=ROOT, *, overwrite=False):
    method = get(config.method)
    root = Path(root).resolve()
    if overwrite:
        target = method.prepare(config, root, overwrite=True)
    else:
        target = method.prepare(config, root)
    if storage.enabled(config) and target is None:
        raise ValueError(
            "Automatic dataset upload requires prepare() to return its artifact directory"
        )
    storage.auto_upload(
        config, "datasets", source=target, overwrite=overwrite, root=root
    )
    return target


def execute(
    config: Config,
    *,
    id=None,
    overwrite=False,
    resume=False,
    root=ROOT,
    trial=None,
    source=None,
    invocation=None,
    display=None,
    _upload=True,
):
    if resume and (overwrite or id is None):
        raise ValueError("Resume requires an ID and cannot be combined with overwrite")
    if trial is not None:
        display = None  # Optuna retains its native output.
    method = get(config.method)
    if not hasattr(method, "run"):
        config.train.require("epochs", "metric", "direction")
    if resume and not getattr(method, "supports_resume", False):
        raise ValueError("This method does not support resume")
    if (
        trial is not None
        and config.search.pruning.strategy != "none"
        and not getattr(method, "supports_pruning", False)
    ):
        raise ValueError("This method does not support pruning")
    id = identifier(id)
    target = directory(config, "runs", id, root)
    with locked(target):
        if resume:
            old = load(target / "config.toml", root=root)
            changed(old, config, {"train.epochs", "device", "platforms", "storage"})
            if not (target / "checkpoint.json").exists():
                raise ValueError("No complete checkpoint is available")
            if not hasattr(method, "run"):
                saved = checkpoint_info(target, "last")
                if config.train.epochs < saved["epoch"]:
                    raise ValueError("Epoch limit cannot precede the saved checkpoint")
                if config.train.epochs == saved["epoch"]:
                    print(f"{id}: epoch limit already reached; no training needed")
                    if _upload:
                        storage.auto_upload(
                            config, "runs", id, overwrite=True, root=root
                        )
                    return id
        else:
            fresh(target, overwrite=overwrite, kind="train")
        run = Run(
            config,
            target,
            resume=resume,
            root=root,
            source=source,
            invocation=invocation,
        )
        try:
            if display:
                display.start(run, supervised=not hasattr(method, "run"))
            run.open_platforms()
            seed_all(config.seed)
            if run.last:
                run.last["state"] = getattr(method, "load_state", load_state)(
                    target / run.last["file"]
                )
            result_source = {"launch": run.launch}
            if hasattr(method, "run"):
                result = method.run(config, run, Path(root).resolve(), trial)
                result_source["kind"] = "custom_final"
            else:
                try:
                    train(
                        method.build(config, Path(root).resolve()),
                        run,
                        trial,
                        on_epoch=display.epoch if display else None,
                        on_start=display.start_epochs if display else None,
                    )
                finally:
                    if display:
                        display.stop_epochs()
                validation = [
                    row for row in history(target) if row["split"] == "validation"
                ]
                best = (max if config.train.direction == "max" else min)(
                    validation, key=lambda row: row["metrics"][config.train.metric]
                )
                result = best["metrics"]
                result_source.update(
                    kind="validation_best",
                    metric_launch=best["launch"],
                    epoch=best["epoch"],
                )
            run.finish("completed", result=result, result_source=result_source)
        except BaseException as exc:
            import optuna

            run.text(traceback.format_exc())
            # Close failure remains fatal; record it rather than disguising it as trial failure.
            try:
                run.finish(
                    "pruned" if isinstance(exc, optuna.TrialPruned) else "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception as close_exc:
                run.platforms = None
                run.finish("failed", error=f"{type(close_exc).__name__}: {close_exc}")
                raise close_exc from exc
            raise
        finally:
            if display:
                display.finish()
    if _upload:
        storage.auto_upload(
            config, "runs", id, overwrite=overwrite or resume, root=root
        )
    return id


def test(
    config: Config,
    *,
    source=None,
    selection="best",
    external=None,
    id=None,
    overwrite=False,
    root=ROOT,
    invocation=None,
    display=None,
    _upload=True,
):
    """Source config owns architecture; evaluation config may select data/protocol/device."""
    if (source is None) == (external is None):
        raise ValueError("Specify exactly one source run or external checkpoint")
    checkpoint_launch = None
    if source:
        if selection == "last":
            raise ValueError(
                "Source testing selects best or Top-K; use external for other checkpoints"
            )
        origin = directory(config, "runs", source, root)
        # Reading a checkpoint must not race a writer's Top-K cleanup or overwrite.
        with locked(origin):
            entry = checkpoint_info(origin, selection)
            checkpoint_launch = entry["launch"]
            original = load(origin / f"config-{checkpoint_launch}.toml", root=root)
            changed(
                original,
                config,
                {
                    "data",
                    "protocol",
                    "device",
                    "platforms",
                    "storage",
                    "name",
                    "tags",
                    "note",
                    "train.epochs",
                },
            )
            # The checkpoint, not the caller's initial config, owns training settings.
            config = config.model_copy(
                update={"train": original.train, "params": original.params}
            )
            file = origin / entry["file"]
            state = getattr(get(config.method), "load_state", load_state)(file)
    else:
        file = path(external, root)
        state = None  # Loaded once below, using the same explicit method loader.
    id = identifier(id)
    target = directory(config, "runs", id, root)
    if source == id or target == file.parent or target in file.parents:
        raise ValueError("Test output must not contain the source checkpoint")
    with locked(target):
        fresh(target, overwrite=overwrite, kind="test")
        run = Run(
            config,
            target,
            kind="test",
            root=root,
            source={
                "run": source,
                "checkpoint": str(file),
                "launch": checkpoint_launch,
            },
            invocation=invocation,
        )
        try:
            if display:
                display.start(run)
            run.open_platforms()
            seed_all(config.seed)
            method = get(config.method)
            if source is None:
                state = getattr(method, "load_state", load_state)(file)
            if hasattr(method, "test"):
                values = method.test(config, run, Path(root).resolve(), file, state)
            else:
                parts = method.build_test(config, Path(root).resolve())
                method.load_checkpoint(parts.model, file, state)
                values, predictions, targets = evaluate(parts, config.device)
                torch.save(
                    {"predictions": predictions, "targets": targets},
                    target / "predictions.pt",
                )
            run.log(values, 0, "test")
            run.finish(
                "completed",
                result=values,
                result_source={"kind": "test", "launch": run.launch},
            )
        except BaseException as exc:
            run.text(traceback.format_exc())
            try:
                run.finish("failed", error=f"{type(exc).__name__}: {exc}")
            except Exception as close_exc:
                run.platforms = None
                run.finish("failed", error=f"{type(close_exc).__name__}: {close_exc}")
                raise close_exc from exc
            raise
        finally:
            if display:
                display.finish()
    if _upload:
        storage.auto_upload(config, "runs", id, overwrite=overwrite, root=root)
    return id
