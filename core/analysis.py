"""Queries and plots read saved results only; no method imports or reruns."""

import csv
import json
import math

import numpy as np
from scipy import stats

from core.config import ROOT, load, path
from core.records import atomic, history, identifier, locked, read_json, snapshot


def list_records(output="experiments", *, root=ROOT, kind=None, **filters):
    result = []
    for folder in ("runs", "studies", "groups"):
        for file in sorted((path(output, root) / folder).glob("*/record.json")):
            row = snapshot(file.parent)
            if kind and row["kind"] != kind:
                continue
            if all(
                (
                    value in row.get("tags", [])
                    if key == "tag"
                    else row.get(key) == value
                )
                for key, value in filters.items()
                if value is not None
            ):
                result.append(row)
    return result


def detail(id, output="experiments", *, root=ROOT, folder="runs"):
    if folder not in ("runs", "studies", "groups"):
        raise ValueError("Unknown record folder")
    target = path(output, root) / folder / identifier(id)
    meta = snapshot(target)
    if folder == "studies":
        import optuna

        from core.tuning import storage

        study = optuna.load_study(study_name=id, storage=storage(target))
        meta["trials"] = [
            {
                "number": trial.number,
                "status": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "run_id": trial.user_attrs.get("run_id"),
                "retry_of": trial.user_attrs.get("retry_of"),
                "error": trial.user_attrs.get("error"),
            }
            for trial in study.trials
        ]
    if folder == "groups":
        meta["member_states"] = {
            member: snapshot(target.parent.parent / "runs" / member)["status"]
            if (target.parent.parent / "runs" / member / "record.json").exists()
            else "pending"
            for member in meta["members"]
        }
    return meta


def annotate(
    id,
    *,
    output="experiments",
    root=ROOT,
    folder="runs",
    tags=None,
    note=None,
    name=None,
):
    if folder not in ("runs", "studies", "groups"):
        raise ValueError("Unknown record folder")
    target = path(output, root) / folder / identifier(id)
    with locked(target):
        meta = read_json(target / "record.json")
        if name is not None:
            meta["name"] = name
        if tags is not None:
            meta["tags"] = tags
        if note is not None:
            meta["note"] = note
        atomic(target / "record.json", meta)


def rows(ids, metric, *, output="experiments", root=ROOT):
    result = []
    for id in ids:
        target = path(output, root) / "runs" / identifier(id)
        meta = read_json(target / "record.json")
        if meta["status"] != "completed":
            raise ValueError(f"{id}: only completed runs can be compared")
        source = meta.get("result_source")
        if not source or source["kind"] not in {
            "validation_best",
            "custom_final",
            "test",
        }:
            raise ValueError(
                f"{id}: missing explicit result source; old records are not automatically migrated"
            )
        config = load(target / f"config-{source['launch']}.toml", root=root)
        value = meta["result"][metric]
        if not math.isfinite(value):
            raise ValueError("Comparison requires finite values")
        result.append(
            {
                "id": id,
                "method": config.method,
                "seed": config.seed,
                "dataset": config.data.name,
                "metric": metric,
                "value": value,
                "protocol": config.protocol.model_dump() | {"kind": meta["kind"]},
                "result_source": source,
                "setup": {"params": config.params, "train": config.train.model_dump()},
            }
        )
    if not result:
        raise ValueError("No records selected")
    return result


def compatible(values):
    if any(
        (row["dataset"], row["protocol"])
        != (values[0]["dataset"], values[0]["protocol"])
        for row in values[1:]
    ):
        raise ValueError(
            "Incompatible dataset/task/split/metric-definition/evaluation identifiers; table-only comparison is allowed"
        )


def table(
    ids, metric, *, destination=None, overwrite=False, output="experiments", root=ROOT
):
    values = rows(ids, metric, output=output, root=root)
    mismatch = any(
        (row["dataset"], row["protocol"])
        != (values[0]["dataset"], values[0]["protocol"])
        for row in values[1:]
    )
    result = [
        row
        | {
            "warning": "INCOMPATIBLE DATASETS/PROTOCOLS — DO NOT AGGREGATE"
            if mismatch
            else ""
        }
        for row in values
    ]
    if destination:
        target = path(destination, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open(
            "w" if overwrite else "x", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
    return result


def summary(ids, metric, *, output="experiments", root=ROOT):
    values = rows(ids, metric, output=output, root=root)
    compatible(values)
    if len({row["seed"] for row in values}) != len(values) or any(
        (row["method"], row["setup"]) != (values[0]["method"], values[0]["setup"])
        for row in values
    ):
        raise ValueError(
            "A multi-seed summary requires one method/setup and distinct seeds"
        )
    scores = [row["value"] for row in values]
    if len(scores) < 2:
        raise ValueError("Sample standard deviation requires at least two seeds")
    return {
        "n": len(scores),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
    }


def paired(left, right, metric, *, test, output="experiments", root=ROOT):
    a, b = (
        rows(left, metric, output=output, root=root),
        rows(right, metric, output=output, root=root),
    )
    compatible(a + b)
    for values in (a, b):
        if len({r["seed"] for r in values}) != len(values) or any(
            (r["method"], r["setup"]) != (values[0]["method"], values[0]["setup"])
            for r in values
        ):
            raise ValueError(
                "Each side must be one method/setup with unique pairing seeds"
            )
    x, y = ({row["seed"]: row["value"] for row in values} for values in (a, b))
    if x.keys() != y.keys() or len(x) < 2 or set(left) & set(right):
        raise ValueError(
            "Provide disjoint runs with at least two exactly matching seed pairs"
        )
    left_values = np.array([x[seed] for seed in sorted(x)])
    right_values = np.array([y[seed] for seed in sorted(x)])
    differences = left_values - right_values
    result = {
        "pairs": len(x),
        "left": {
            "mean": float(left_values.mean()),
            "std": float(left_values.std(ddof=1)),
        },
        "right": {
            "mean": float(right_values.mean()),
            "std": float(right_values.std(ddof=1)),
        },
        "mean_difference": float(differences.mean()),
        "test": test,
        "alternative": "two-sided",
        "p_value": None,
        "status": "ok",
    }
    if test == "ttest":
        if np.all(differences == differences[0]):
            return result | {
                "status": "not_computed",
                "reason": "Paired differences have zero variance",
            }
        statistic = stats.ttest_rel(left_values, right_values, alternative="two-sided")
    elif test == "wilcoxon":
        if not np.any(differences):
            return result | {
                "status": "not_computed",
                "reason": "All paired differences are zero (zero_method='wilcox')",
            }
        statistic = stats.wilcoxon(
            differences, zero_method="wilcox", alternative="two-sided", method="auto"
        )
    else:
        raise ValueError("Choose ttest or wilcoxon")
    if not math.isfinite(statistic.pvalue):
        raise ValueError("Statistical test returned an undefined p-value")
    result["p_value"] = float(statistic.pvalue)
    return result


def curves(
    ids,
    metric,
    *,
    split,
    output="experiments",
    root=ROOT,
    title=None,
    figsize=(6, 4),
    xlabel="Epoch",
    ylabel=None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for id in ids:
        values = [
            row
            for row in history(path(output, root) / "runs" / identifier(id))
            if row["split"] == split
        ]
        if not values:
            raise ValueError(f"No committed {split} history for {id}")
        ax.plot(
            [row["epoch"] if "epoch" in row else row["step"] for row in values],
            [row["metrics"][metric] for row in values],
            label=id,
        )
    ax.set(xlabel=xlabel, ylabel=ylabel or metric, title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def classification(
    targets, predictions, *, labels=None, title="Confusion Matrix", figsize=(5, 4)
):
    import matplotlib.pyplot as plt

    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.ndim != 1 or predictions.shape != targets.shape or not len(targets):
        raise ValueError("Provide nonempty equally-sized class-label vectors")
    labels = (
        list(labels)
        if labels is not None
        else np.unique(np.concatenate([targets, predictions])).tolist()
    )
    index = {value: i for i, value in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, predicted in zip(targets, predictions, strict=True):
        matrix[index[true], index[predicted]] += 1
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    return fig


def save_figure(fig, destination, *, dpi=300, overwrite=False, root=ROOT):
    target = path(destination, root)
    if target.suffix.lower() not in (".png", ".pdf"):
        raise ValueError("Use PNG or PDF")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb" if overwrite else "xb") as file:
        fig.savefig(file, format=target.suffix[1:], dpi=dpi)
    return target


def pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def save_json(result, destination, *, overwrite=False, root=ROOT):
    """Export the complete result, never the formatted terminal view."""
    target = path(destination, root)
    content = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w" if overwrite else "x", encoding="utf-8") as file:
        file.write(content)
    return target
