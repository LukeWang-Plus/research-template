"""Tiny prepared datasets; training never creates or splits them implicitly."""

import json

import torch

from core.config import path


def location(config, root):
    if config.data.path is None:
        raise ValueError("The toy examples require data.path")
    return path(config.data.path, root)


def prepare(config, root, ragged=False):
    target = location(config, root)
    target.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(17)
    for split, size in [("train", 48), ("validation", 20), ("test", 20)]:
        if ragged:
            rows = []
            for index in range(size):
                values = torch.randn(2 + index % 5, generator=generator).tolist()
                rows.append(
                    {
                        "observations": values,
                        "target": 2 * sum(values) / len(values) + 0.5,
                    }
                )
            (target / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
        else:
            x = torch.randn(size, 3, generator=generator)
            y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
            torch.save({"x": x, "y": y}, target / f"{split}.pt")
    return target


def tensors(config, root, split):
    data = torch.load(location(config, root) / f"{split}.pt", weights_only=True)
    return torch.utils.data.TensorDataset(data["x"], data["y"])


def ragged(config, root, split):
    return json.loads(
        (location(config, root) / f"{split}.json").read_text(encoding="utf-8")
    )
