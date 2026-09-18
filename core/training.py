"""Optional supervised loop. Research code owns all data and model construction."""

from dataclasses import dataclass
from typing import Callable

import torch

from core.records import numbers, restore_rng, rng_state


@dataclass
class Training:
    model: torch.nn.Module
    train: object
    validation: object
    optimizer: torch.optim.Optimizer
    batch: Callable
    metrics: Callable
    scheduler: object = None
    eval_batch: Callable | None = None
    # Explicitly supplied stateful components, e.g. a loader's independent Generator.
    stateful: dict | None = None


@dataclass
class Evaluation:
    model: torch.nn.Module
    data: object
    batch: Callable
    metrics: Callable


def loss_weight(result):
    loss = result["loss"]
    weight = float(result["weight"])
    if (
        not isinstance(loss, torch.Tensor)
        or loss.ndim != 0
        or not torch.isfinite(loss).item()
    ):
        raise ValueError("loss must be a finite scalar Tensor")
    if not 0 < weight < float("inf"):
        raise ValueError("loss weight must be positive and finite")
    return loss, weight


def evaluate(parts: Evaluation, device="cpu"):
    model = parts.model.to(device)
    model.eval()
    predictions, targets = [], []
    total = denominator = 0.0
    loss_presence = set()
    with torch.no_grad():
        for batch in parts.data:
            result = parts.batch(model, batch, device)
            loss_presence.add("loss" in result)
            if "loss" in result:
                loss, weight = loss_weight(result)
                total += loss.item() * weight
                denominator += weight
            pred, target = result["predictions"], result["targets"]
            if (
                not isinstance(pred, torch.Tensor)
                or not isinstance(target, torch.Tensor)
                or pred.ndim == 0
                or target.ndim == 0
                or pred.shape[0] != target.shape[0]
            ):
                raise ValueError(
                    "Evaluation requires tensors with matching batch dimension"
                )
            predictions.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
    if not predictions or len(loss_presence) > 1:
        raise ValueError("Empty evaluation loader or inconsistent loss presence")
    pred, target = torch.cat(predictions), torch.cat(targets)
    values = numbers(parts.metrics(pred, target))
    if denominator:
        if "loss" in values:
            raise ValueError("'loss' is reserved for the weighted batch loss")
        values["loss"] = total / denominator
    return values, pred, target


def train(parts: Training, run, trial=None):
    config = run.config
    config.train.require("epochs", "metric", "direction")
    model = parts.model.to(config.device)
    epoch = bad = 0
    best = None
    if run.last:
        state = run.last["state"]
        model.load_state_dict(state["model"])
        parts.optimizer.load_state_dict(state["optimizer"])
        if parts.scheduler is not None:
            parts.scheduler.load_state_dict(state["scheduler"])
        for name, component in (parts.stateful or {}).items():
            component.load_state_dict(state["components"][name])
        restore_rng(state["rng"])
        epoch = run.last["epoch"]
        best, bad = state["early_stop"]["best"], state["early_stop"]["bad"]
    values = run.last["state"].get("validation") if run.last else None
    for epoch in range(epoch + 1, config.train.epochs + 1):
        if config.train.patience is not None and bad >= config.train.patience:
            break
        model.train()
        total = denominator = 0.0
        for batch in parts.train:
            parts.optimizer.zero_grad(set_to_none=True)
            result = parts.batch(model, batch, config.device)
            loss, weight = loss_weight(result)
            loss.backward()
            parts.optimizer.step()
            total += loss.item() * weight
            denominator += weight
        if not denominator:
            raise ValueError("Empty training loader")
        run.log({"loss": total / denominator}, epoch, "train")
        values, _, _ = evaluate(
            Evaluation(
                model, parts.validation, parts.eval_batch or parts.batch, parts.metrics
            ),
            config.device,
        )
        score = values[config.train.metric]
        run.log(values, epoch)
        if parts.scheduler is not None:
            if isinstance(parts.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                parts.scheduler.step(score)
            else:
                parts.scheduler.step()
        improvement = (
            best is None
            or (score - best if config.train.direction == "max" else best - score)
            > config.train.min_delta
        )
        if improvement:
            best, bad = score, 0
        else:
            bad += 1
        state = {
            "model": model.state_dict(),
            "optimizer": parts.optimizer.state_dict(),
            "scheduler": parts.scheduler.state_dict() if parts.scheduler else None,
            "components": {
                name: obj.state_dict() for name, obj in (parts.stateful or {}).items()
            },
            "early_stop": {"best": best, "bad": bad},
            "rng": rng_state(),
            "validation": values,
        }
        run.save_checkpoint(state, epoch, score)
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned(f"pruned at epoch {epoch}")
    if values is None:
        raise ValueError("No complete epoch available")
    return values
