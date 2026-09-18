"""Minimal supervised classification; also usable entirely in memory."""

import torch
from torch.utils.data import DataLoader

from core.training import Evaluation, Training
from dataio import toy

supports_resume = True
supports_pruning = True


def prepare(config, root):
    return toy.prepare(config, root)


def model(hidden=0):
    if hidden:
        return torch.nn.Sequential(
            torch.nn.Linear(3, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2)
        )
    return torch.nn.Linear(3, 2)


def batch(model, data, device):
    x, y = (tensor.to(device) for tensor in data)
    logits = model(x)
    return {
        "loss": torch.nn.functional.cross_entropy(logits, y),
        "weight": len(y),
        "predictions": logits,
        "targets": y,
    }


def metrics(predictions, targets):
    return {"accuracy": (predictions.argmax(1) == targets).float().mean().item()}


def options(config):
    params = {
        "lr": 0.1,
        "batch_size": 8,
        "hidden": 0,
        "scheduler_gamma": 1.0,
    } | config.params
    unknown = params.keys() - {"lr", "batch_size", "hidden", "scheduler_gamma"}
    if unknown or config.data.params:
        raise ValueError(
            f"Unsupported example parameters: {unknown or config.data.params}"
        )
    return params


def build(config, root):
    p = options(config)
    network = model(p["hidden"]).to(config.device)
    optimizer = torch.optim.SGD(network.parameters(), lr=p["lr"], momentum=0.9)
    return Training(
        network,
        DataLoader(
            toy.tensors(config, root, "train"), batch_size=p["batch_size"], shuffle=True
        ),
        DataLoader(toy.tensors(config, root, "validation"), batch_size=p["batch_size"]),
        optimizer,
        batch,
        metrics,
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=p["scheduler_gamma"]),
    )


def build_test(config, root):
    p = options(config)
    return Evaluation(
        model(p["hidden"]),
        DataLoader(toy.tensors(config, root, "test"), batch_size=p["batch_size"]),
        batch,
        metrics,
    )


def load_checkpoint(model, file, state=None):
    if state is None:
        state = torch.load(file, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
