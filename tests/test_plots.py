"""Saved-result semantics, local styles, and native figure/export integration."""

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from typer.testing import CliRunner

from core import analysis, plotting
from core.cli import app
from core.config import dumps
from core.records import atomic
from tests.test_workflow import cfg


@pytest.fixture(autouse=True)
def plot_context():
    with mpl.rc_context():
        yield
        plt.close("all")


def saved_runs(root, extra=0):
    runs = [
        ("a1", 1, 0.6, 0),
        ("a2", 2, 0.8, 0),
        ("b1", 1, 0.8, 4),
        ("b2", 2, 0.9, 4),
        ("single", 3, 0.7, 8),
    ]
    runs.extend((f"extra-{i}", i + 4, 0.5, 0) for i in range(extra))
    for name, seed, score, hidden in runs:
        c = cfg(root, seed=seed)
        c.params["hidden"] = hidden
        directory = root / "experiments/runs" / name
        atomic(directory / "config-1.toml", dumps(c))
        atomic(
            directory / "record.json",
            {
                "id": name,
                "kind": "train",
                "status": "completed",
                "result": {"accuracy": score},
                "result_source": {"kind": "custom_final", "launch": 1},
            },
        )
        (directory / "metrics-1.jsonl").write_text(
            '{"epoch":1,"split":"validation","metrics":{"accuracy":0.25}}\n'
            '{"epoch":2,"split":"validation","metrics":{"accuracy":0.5}}\n'
        )


def test_bars_statistics_and_comparability(tmp_path):
    saved_runs(tmp_path)
    series = [["a1", "a2"], ["b1", "b2"], ["single"]]
    before = {
        p: p.read_bytes() for p in (tmp_path / "experiments").rglob("*") if p.is_file()
    }
    fig = plotting.bars(series, "accuracy", labels=["A", "B", "Single"], root=tmp_path)
    ax = fig.axes[0]
    np.testing.assert_allclose([p.get_height() for p in ax.patches], [0.7, 0.85, 0.7])
    error_segments = [collection.get_segments()[0] for collection in ax.collections]
    assert len(error_segments) == 2  # The single result has no error bar.
    for segment, ids in zip(error_segments, series[:2], strict=True):
        stats = analysis.summary(ids, "accuracy", root=tmp_path)
        np.testing.assert_allclose(
            segment[:, 1], [stats["mean"] - stats["std"], stats["mean"] + stats["std"]]
        )
    assert ax.get_ylim()[0] == 0
    assert [p.get_facecolor() for p in ax.patches] == [
        mpl.colors.to_rgba(color) for color in cc.glasbey[:3]
    ]
    assert [label.get_text() for label in ax.get_xticklabels()] == ["A", "B", "Single"]
    assert all(label.get_rotation() == 0 for label in ax.get_xticklabels())
    assert not fig.texts  # No automatic explanatory text inside the figure.
    plt.close(fig)
    for bad in ([], [[]], [["a1", "a1"]], [["a1"], ["a1"]], [["a1", "b2"]]):
        with pytest.raises(ValueError):
            plotting.bars(bad, "accuracy", root=tmp_path)
    with pytest.raises(ValueError, match="one display label"):
        plotting.bars(series, "accuracy", labels=["A"], root=tmp_path)
    assert all(p.read_bytes() == data for p, data in before.items())
    # Cross-bar compatibility is checked even when each bar has just one run.
    c = cfg(tmp_path, seed=2)
    c.data.name = "different"
    atomic(tmp_path / "experiments/runs/b2/config-1.toml", dumps(c))
    with pytest.raises(ValueError, match="Incompatible"):
        plotting.bars([["a1"], ["b2"]], "accuracy", root=tmp_path)
    c.data.name = "toy-classification"
    c.seed = 1
    atomic(tmp_path / "experiments/runs/a2/config-1.toml", dumps(c))
    with pytest.raises(ValueError, match="distinct seeds"):
        plotting.bars([["a1", "a2"]], "accuracy", root=tmp_path)


def test_local_styles_cli_and_export(tmp_path):
    from PIL import Image
    from tueplots import bundles

    saved_runs(tmp_path, extra=12)
    before = dict(mpl.rcParams)
    fig = plotting.curves(
        ["a1", "b1"],
        "accuracy",
        split="validation",
        labels=[r"A $\alpha$", r"B $\beta$"],
        root=tmp_path,
    )
    np.testing.assert_equal(fig.axes[0].lines[0].get_ydata(), [0.25, 0.5])
    assert [line.get_label() for line in fig.axes[0].lines] == [
        r"A $\alpha$",
        r"B $\beta$",
    ]
    np.testing.assert_allclose(fig.get_size_inches(), (6, 4))
    ax = fig.axes[0]
    science = plt.style.library["science"]
    assert ax.lines[0].get_linewidth() == science["lines.linewidth"]
    assert ax.xaxis.get_major_ticks()[0].tick2line.get_visible() == science["xtick.top"]
    assert isinstance(ax.xaxis.get_minor_locator(), mpl.ticker.AutoMinorLocator)
    assert [line.get_color() for line in ax.lines] == cc.glasbey[:2]
    assert ax.xaxis.label.get_usetex() is False
    base_png = plotting.save_figure(fig, "base.png", root=tmp_path)
    with Image.open(base_png) as image:
        assert image.size == (1800, 1200)
    base_pdf = plotting.save_figure(fig, "base.pdf", root=tmp_path)
    assert b"/FontFile2" in base_pdf.read_bytes()
    assert dict(mpl.rcParams) == before
    plt.close(fig)

    ids = [f"extra-{i}" for i in range(12)]
    for style in (None, {"axes.prop_cycle": plt.cycler(color=["black", "gray"])}):
        curve = plotting.curves(
            ids, "accuracy", split="validation", style=style, root=tmp_path
        )
        bar = plotting.bars(
            [[id] for id in ids], "accuracy", style=style, root=tmp_path
        )
        colors = cc.glasbey[:12] if style is None else ["black", "gray"] * 6
        assert [line.get_color() for line in curve.axes[0].lines] == colors
        assert [p.get_facecolor() for p in bar.axes[0].patches] == [
            mpl.colors.to_rgba(color) for color in colors
        ]
        plt.close(curve)
        plt.close(bar)

    settings = bundles.icml2024(usetex=False) | {"font.serif": ["DejaVu Serif"]}
    fig = plotting.curves(
        ["a1"], "accuracy", split="validation", style=settings, root=tmp_path
    )
    np.testing.assert_allclose(fig.get_size_inches(), settings["figure.figsize"])
    assert fig.axes[0].xaxis.label.get_fontsize() == settings["axes.labelsize"]
    target = plotting.save_figure(
        fig,
        "plot.png",
        style=settings | {"savefig.bbox": None, "savefig.dpi": 80},
        root=tmp_path,
        dpi=100,
    )
    with Image.open(target) as image:
        assert image.size == tuple(int(v * 100) for v in settings["figure.figsize"])
    plotting.save_figure(fig, "plot.pdf", style=settings, root=tmp_path)
    assert (tmp_path / "plot.pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(FileExistsError):
        plotting.save_figure(fig, "plot.pdf", root=tmp_path)
    assert dict(mpl.rcParams) == before
    plt.close(fig)
    cli = CliRunner()
    for args in (
        [
            "plot",
            "a1",
            "--split",
            "validation",
            "--label",
            r"Baseline $\alpha$",
            "--destination",
            "cli.png",
        ],
        [
            "bars",
            "--runs",
            "a1 a2",
            "--runs",
            "b1 b2",
            "--label",
            "A",
            "--label",
            "B",
            "--destination",
            "cli.pdf",
        ],
    ):
        args += [
            "--metric",
            "accuracy",
            "--root",
            str(tmp_path),
            "--width",
            "3.4",
            "--height",
            "2.5",
            "--ylabel",
            "Accuracy",
        ]
        result = cli.invoke(app, args)
        assert result.exit_code == 0, result.exception
        assert "Saved figure:" in result.output
        assert isinstance(cli.invoke(app, args).exception, FileExistsError)
        assert cli.invoke(app, args + ["--overwrite"]).exit_code == 0
    assert dict(mpl.rcParams) == before
    plt.close("all")


def test_confusion_counts_contrast_and_palette():
    before = dict(mpl.rcParams)
    fig = plotting.classification([0, 0, 1, 1, 1], [0, 1, 1, 1, 1], labels=[1, 0])
    ax = fig.axes[0]
    np.testing.assert_allclose(fig.get_size_inches(), (6, 4))
    np.testing.assert_allclose(
        ax.images[0].get_cmap()(np.linspace(0, 1, 5)),
        cc.cm["CET_CBL2_r"](np.linspace(0, 1, 5)),
    )
    np.testing.assert_equal(ax.images[0].get_array(), [[3, 0], [1, 1]])
    assert ax.images[0].get_clim()[0] == 0
    assert {text.get_color() for text in ax.texts} == {"black", "white"}
    assert fig.axes[1].get_ylabel() == "Count"
    assert all(label.get_rotation() == 0 for label in ax.get_xticklabels())
    assert ax.texts[0].get_usetex() is False
    assert dict(mpl.rcParams) == before
    plt.close(fig)
    for labels in ([0], [0, 0, 1]):
        with pytest.raises(ValueError, match="Labels must"):
            plotting.classification([0, 1], [0, 1], labels=labels)

    fig = plotting.classification(
        [0, 1],
        [0, 1],
        style={"figure.constrained_layout.use": True},
    )
    assert isinstance(
        fig.get_layout_engine(), mpl.layout_engine.ConstrainedLayoutEngine
    )
    assert not fig.get_layout_engine()._compress


def test_external_settings_override_defaults(tmp_path):
    from io import BytesIO

    from PIL import Image

    saved_runs(tmp_path)
    external = {
        "figure.figsize": (4, 3),
        "axes.prop_cycle": plt.cycler(color=["black", "gray"]),
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": True,
        "axes.ymargin": 0.3,
        "errorbar.capsize": 7,
        "lines.linewidth": 2.5,
        "image.cmap": "Blues",
        "figure.constrained_layout.use": False,
        "figure.autolayout": True,
        "savefig.bbox": "tight",
        "savefig.dpi": 80,
        "pdf.fonttype": 3,
    }
    before = dict(mpl.rcParams)
    with mpl.rc_context({"figure.figsize": (9, 7)}):
        default = plotting.curves(["a1"], "accuracy", split="validation", root=tmp_path)
        np.testing.assert_allclose(default.get_size_inches(), (6, 4))
        assert mpl.rcParams["figure.figsize"] == [9, 7]

    curve = plotting.curves(
        ["a1"], "accuracy", split="validation", style=external, root=tmp_path
    )
    bar = plotting.bars([["a1", "a2"]], "accuracy", style=external, root=tmp_path)
    matrix = plotting.classification([0, 1], [0, 1], style=external)
    for fig in (curve, bar, matrix):
        ax = fig.axes[0]
        np.testing.assert_allclose(fig.get_size_inches(), (4, 3))
        assert isinstance(fig.get_layout_engine(), mpl.layout_engine.TightLayoutEngine)
        assert ax.spines["top"].get_visible() and ax.spines["right"].get_visible()
        assert isinstance(ax.xaxis.get_minor_locator(), mpl.ticker.AutoMinorLocator)
        assert isinstance(ax.yaxis.get_minor_locator(), mpl.ticker.AutoMinorLocator)
        assert any(line.get_visible() for line in ax.get_xgridlines())
        assert ax.xaxis.get_major_ticks()[0].tick2line.get_visible()
    assert curve.axes[0].lines[0].get_color() == "black"
    assert bar.axes[0].patches[0].get_facecolor() == mpl.colors.to_rgba("black")
    assert bar.axes[0].margins()[1] == 0.3
    assert bar.axes[0].containers[0].lines[1][0].get_markersize() == 14
    assert matrix.axes[0].images[0].get_cmap().name == "Blues"

    # Pass export overrides explicitly; drawing does not retain them.
    expected = BytesIO()
    with plt.style.context(["science", "no-latex", external]):
        curve.savefig(expected, format="png")
    target = plotting.save_figure(curve, "external.png", style=external, root=tmp_path)
    with Image.open(expected) as native, Image.open(target) as actual:
        assert actual.size == native.size
    pdf = plotting.save_figure(curve, "external.pdf", style=external, root=tmp_path)
    assert b"/Subtype /Type3" in pdf.read_bytes()
    base_pdf = plotting.save_figure(curve, "base.pdf", root=tmp_path)
    assert b"/Subtype /Type3" not in base_pdf.read_bytes()
    assert curve.axes[0].lines[0].get_color() == "black"
    np.testing.assert_allclose(curve.get_size_inches(), (4, 3))

    explicit = plotting.curves(
        ["a1"],
        "accuracy",
        split="validation",
        root=tmp_path,
        style=[
            external,
            {"axes.grid": False, "axes.prop_cycle": plt.cycler(color=["red"])},
        ],
        figsize=(5, 2),
        layout="none",
    )
    np.testing.assert_allclose(explicit.get_size_inches(), (5, 2))
    assert explicit.get_layout_engine() is None
    assert explicit.axes[0].lines[0].get_color() == "red"
    assert not any(line.get_visible() for line in explicit.axes[0].get_xgridlines())
    explicit.axes[0].lines[0].set_color("green")
    assert explicit.axes[0].lines[0].get_color() == "green"
    matrix = plotting.classification([0, 1], [0, 1], style=external, cmap="Greys")
    assert matrix.axes[0].images[0].get_cmap().name == "Greys"
    assert dict(mpl.rcParams) == before
