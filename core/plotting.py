"""Local figure styles and exports; saved results are never modified or rerun."""

import numpy as np

from core.analysis import _summary, compatible, rows
from core.config import ROOT, path
from core.records import history, identifier


def _style(style):
    import colorcet as cc
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401 -- registers native styles

    base = [
        "science",
        "no-latex",
        {
            "axes.prop_cycle": plt.cycler(color=cc.glasbey),
            "image.cmap": "cet_CET_CBL2_r",
            "figure.figsize": (6, 4),
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "savefig.bbox": None,
        },
    ]
    if style is not None:
        base.extend(style if isinstance(style, (list, tuple)) else [style])
    return base


def _labels(labels, defaults):
    labels = list(defaults if labels is None else labels)
    if not labels or len(labels) != len(defaults):
        raise ValueError("Provide one display label per plotted series")
    return labels


def curves(
    ids,
    metric,
    *,
    split,
    output="experiments",
    root=ROOT,
    title=None,
    figsize=None,
    xlabel="Epoch",
    ylabel=None,
    labels=None,
    style=None,
    layout=None,
):
    """Plot committed histories without smoothing or aggregating runs."""
    import matplotlib.pyplot as plt

    ids = list(ids)
    names = _labels(labels, ids)
    histories = []
    for id in ids:
        values = [
            row
            for row in history(path(output, root) / "runs" / identifier(id))
            if row["split"] == split
        ]
        if not values:
            raise ValueError(f"No committed {split} history for {id}")
        histories.append(
            (
                [row["epoch"] if "epoch" in row else row["step"] for row in values],
                [row["metrics"][metric] for row in values],
            )
        )
    with plt.style.context(_style(style)):
        fig, ax = plt.subplots(figsize=figsize, layout=layout)
        for (x, y), name in zip(histories, names, strict=True):
            ax.plot(x, y, label=name)
        ax.set(xlabel=xlabel, ylabel=metric if ylabel is None else ylabel, title=title)
        ax.legend()
    return fig


def bars(
    series,
    metric,
    *,
    output="experiments",
    root=ROOT,
    labels=None,
    title=None,
    figsize=None,
    xlabel=None,
    ylabel=None,
    style=None,
    layout=None,
    width=0.65,
):
    """One explicit run collection per bar; mean ± sample SD, or a single value."""
    import matplotlib.pyplot as plt

    series = [list(ids) for ids in series]
    if not series or any(not ids for ids in series):
        raise ValueError("Provide nonempty run collections, one per bar")
    all_ids = [id for ids in series for id in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Each run may appear only once in a bar comparison")
    names = _labels(labels, [", ".join(ids) for ids in series])
    values = [rows(ids, metric, output=output, root=root) for ids in series]
    compatible([row for group in values for row in group])
    summaries = [
        _summary(group) if len(group) > 1 else {"mean": group[0]["value"], "std": None}
        for group in values
    ]
    with plt.style.context(_style(style)):
        fig, ax = plt.subplots(figsize=figsize, layout=layout)
        for i, result in enumerate(summaries):
            ax.bar(i, result["mean"], width=width, yerr=result["std"])
        ax.set(
            xticks=range(len(names)),
            xticklabels=names,
            xlabel=xlabel,
            ylabel=metric if ylabel is None else ylabel,
            title=title,
        )
        lower, upper = ax.get_ylim()
        ax.set_ylim(min(0, lower), max(0, upper))
    return fig


def classification(
    targets,
    predictions,
    *,
    labels=None,
    title="Confusion Matrix",
    figsize=None,
    cmap=None,
    style=None,
    layout=None,
):
    """Count class labels, with contrast-aware annotations; no normalization."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.ndim != 1 or predictions.shape != targets.shape or not len(targets):
        raise ValueError("Provide nonempty equally-sized class-label vectors")
    observed = np.unique(np.concatenate([targets, predictions])).tolist()
    labels = list(labels) if labels is not None else observed
    if len(set(labels)) != len(labels) or not set(observed) <= set(labels):
        raise ValueError("Labels must be unique and include every observed class")
    index = {value: i for i, value in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, predicted in zip(targets, predictions, strict=True):
        matrix[index[true], index[predicted]] += 1
    with plt.style.context(_style(style)):
        fig, ax = plt.subplots(figsize=figsize, layout=layout)
        image = ax.imshow(matrix, cmap=cmap, vmin=0)
        colorbar = fig.colorbar(image, ax=ax, label="Count")
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
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
            rgb = np.asarray(image.cmap(image.norm(value))[:3])
            linear = np.where(
                rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
            )
            luminance = linear @ [0.2126, 0.7152, 0.0722]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="black" if luminance > 0.179 else "white",
            )
    return fig


def save_figure(fig, destination, *, style=None, dpi=None, overwrite=False, root=ROOT):
    """Apply local export settings; pass drawing overrides again when saving."""
    import matplotlib.pyplot as plt

    target = path(destination, root)
    if target.suffix.lower() not in (".png", ".pdf"):
        raise ValueError("Use PNG or PDF")
    with plt.style.context(_style(style)):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb" if overwrite else "xb") as file:
            fig.savefig(file, format=target.suffix[1:], dpi=dpi)
    return target
