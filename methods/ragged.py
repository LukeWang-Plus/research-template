"""Variable-length dictionaries and a fully custom loop, without Trainer hooks."""

from dataclasses import dataclass

import torch

from core.records import checkpoint, restore_rng, rng_state
from dataio import toy

supports_resume = True
supports_pruning = True


@dataclass
class Memory:
    epoch: int = 0


def load_state(file):
    # Only load trusted checkpoints: Memory requires Python object deserialization.
    return torch.load(file, map_location="cpu", weights_only=False)


def suggest(trial, config):
    params = config["params"]
    params["optimizer"] = trial.suggest_categorical("params.optimizer", ["sgd", "adam"])
    params.pop("momentum", None)
    if params["optimizer"] == "sgd":
        params["momentum"] = trial.suggest_float("params.momentum", 0.0, 0.9)
    return config


def objective(config, run_directory, trial):
    state = load_state(checkpoint(run_directory, "best"))
    parameter_count = sum(value.numel() for value in state["model"].values())
    # Penalize model size across trials; checkpointing and pruning use MSE.
    return state["validation"]["mse"] + 0.001 * parameter_count


def prepare(config, root):
    return toy.prepare(config, root, ragged=True)


def network(config):
    if (
        config.params.keys() - {"lr", "optimizer", "momentum", "hidden"}
        or config.data.params
    ):
        raise ValueError("Unsupported ragged parameters")
    optimizer = config.params.get("optimizer", "sgd")
    if optimizer not in {"sgd", "adam"}:
        raise ValueError("optimizer must be sgd or adam")
    if optimizer != "sgd" and "momentum" in config.params:
        raise ValueError("momentum is only supported with sgd")
    hidden = config.params.get("hidden", 0)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)
        )
        if hidden
        else torch.nn.Linear(1, 1)
    )
    return model.to(config.device)


def prediction(model, row, device):
    values = torch.tensor(row["observations"], device=device)
    return model(values.mean().reshape(1, 1)).squeeze()


def evaluate(model, rows, device):
    if not rows:
        raise ValueError("Empty evaluation data")
    model.eval()
    with torch.no_grad():
        predictions = [prediction(model, row, device).item() for row in rows]
    mse = sum(
        (pred - row["target"]) ** 2 for pred, row in zip(predictions, rows, strict=True)
    ) / len(rows)
    return {"mse": mse}, predictions


def run(config, run, root, trial=None):
    config.train.require("epochs", "metric", "direction")
    if run.last and config.train.epochs < run.last["epoch"]:
        raise ValueError("Epoch limit cannot precede the saved checkpoint")
    model = network(config)
    lr = config.params.get("lr", 0.05)
    optimizer = (
        torch.optim.SGD(
            model.parameters(), lr=lr, momentum=config.params.get("momentum", 0.0)
        )
        if config.params.get("optimizer", "sgd") == "sgd"
        else torch.optim.Adam(model.parameters(), lr=lr)
    )
    train_data, validation = (
        toy.ragged(config, root, "train"),
        toy.ragged(config, root, "validation"),
    )
    memory = Memory()
    bad = 0
    best = None
    if run.last:
        state = run.last["state"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        memory = state["memory"]
        best, bad = state["best"], state["bad"]
    metrics = run.last["state"]["validation"] if run.last else None
    for epoch in range(memory.epoch + 1, config.train.epochs + 1):
        if config.train.patience is not None and bad >= config.train.patience:
            break
        if not train_data:
            raise ValueError("Empty training data")
        model.train()
        total = 0.0
        for row in train_data:
            optimizer.zero_grad(set_to_none=True)
            loss = (prediction(model, row, config.device) - row["target"]) ** 2
            loss.backward()
            optimizer.step()
            total += loss.item()
        metrics, _ = evaluate(model, validation, config.device)
        score = metrics[config.train.metric]
        run.log({"loss": total / len(train_data)}, epoch, "train")
        run.log(metrics, epoch)
        improved = (
            best is None
            or (score - best if config.train.direction == "max" else best - score)
            > config.train.min_delta
        )
        best, bad = (score, 0) if improved else (best, bad + 1)
        memory.epoch = epoch
        run.save_checkpoint(
            {
                "memory": memory,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "best": best,
                "bad": bad,
                "validation": metrics,
            },
            epoch,
            score,
        )
        if trial:
            # This custom loop uses completed epochs (starting at 1) as pruning steps.
            trial.report(score, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
    return metrics


def test(config, run, root, file, state=None):
    model = network(config)
    state = state if state is not None else load_state(file)
    model.load_state_dict(state["model"])
    metrics, predictions = evaluate(
        model, toy.ragged(config, root, "test"), config.device
    )
    from core.records import atomic

    atomic(run.directory / "predictions.json", predictions)
    return metrics
