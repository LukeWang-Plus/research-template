"""Thin CLI over synchronous Python functions."""

import sys
from pathlib import Path
from typing import Annotated

import typer

from core import analysis, display, groups, runner, tuning
from core.config import ROOT, load, path

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
group_app = typer.Typer(no_args_is_help=True)
app.add_typer(group_app, name="group")
Root = Annotated[Path, typer.Option(help="Project root for relative paths")]
Overrides = Annotated[
    list[str] | None,
    typer.Option("--set", help='TOML override, e.g. params.lr=0.01 or device="cuda:0"'),
]


def save_result(result, destination, overwrite, root):
    if destination is not None:
        saved = analysis.save_json(result, destination, overwrite=overwrite, root=root)
        display.message(f"Saved JSON: {saved}")


def config(file, overrides, root):
    return load(file, overrides or [], root)


@app.command()
def prepare(file: Path, root: Root = ROOT, set: Overrides = None):
    """Explicitly prepare data; existing prepared data is never silently rebuilt."""
    display.message(f"Prepared data: {runner.prepare(config(file, set, root), root)}")


@app.command()
def train(
    file: Path,
    id: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
    root: Root = ROOT,
    set: Overrides = None,
):
    """Train or resume a run (resume and overwrite are different operations)."""
    runner.execute(
        config(file, set, root),
        id=id,
        resume=resume,
        overwrite=overwrite,
        root=root,
        invocation=sys.argv,
        display=display.RunDisplay(),
    )


@app.command()
def test(
    file: Path,
    source: str | None = None,
    checkpoint: str = "best",
    external: str | None = None,
    id: str | None = None,
    overwrite: bool = False,
    root: Root = ROOT,
    set: Overrides = None,
):
    """Create an independent test result from a source run or external checkpoint."""
    runner.test(
        config(file, set, root),
        source=source,
        selection=checkpoint,
        external=external,
        id=id,
        overwrite=overwrite,
        root=root,
        invocation=sys.argv,
        display=display.RunDisplay(),
    )


@app.command()
def tune(
    file: Path,
    trials: Annotated[int, typer.Option(min=1)],
    id: str | None = None,
    continuation: bool = False,
    retry: int | None = None,
    root: Root = ROOT,
    set: Overrides = None,
):
    """Start/continue a study; trials is the additional budget for this invocation."""
    typer.echo(
        tuning.tune(
            config(file, set, root),
            trials=trials,
            id=id,
            continuation=continuation,
            retry=retry,
            root=root,
        )
    )


@app.command("export")
def export_trial(
    file: Path,
    study: str,
    destination: Path,
    trial: int | None = None,
    overwrite: bool = False,
    root: Root = ROOT,
):
    saved = tuning.export(
        load(file, root=root),
        study,
        destination,
        trial_number=trial,
        overwrite=overwrite,
        root=root,
    )
    display.message(f"Saved TOML: {saved}")


@group_app.command("create")
def group_create(
    files: Annotated[list[Path], typer.Argument()],
    seeds: Annotated[list[int], typer.Option("--seed")],
    id: str | None = None,
    root: Root = ROOT,
):
    created = groups.create(
        [load(file, root=root) for file in files], seeds, id=id, root=root
    )
    display.message(f"Created group: {created}")


@group_app.command("run")
def group_run(file: Path, id: str, action: str | None = None, root: Root = ROOT):
    """Continue serially; --action resume/rerun handles failed members explicitly."""
    completed = groups.run_group(
        load(file, root=root),
        id,
        action=action,
        root=root,
        display=display.RunDisplay(),
    )
    display.message(f"Completed group: {completed}")


@group_app.command("test")
def group_test(
    file: Path,
    id: str,
    members: Annotated[list[str], typer.Option("--member")],
    root: Root = ROOT,
):
    results = groups.test_members(
        load(file, root=root), id, members, root=root, display=display.RunDisplay()
    )
    display.table("Test runs", ("Run ID",), ((item,) for item in results))


@app.command("list")
def list_runs(
    output: str = "experiments",
    kind: str | None = None,
    method: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    status: str | None = None,
    tag: str | None = None,
    save: Path | None = None,
    overwrite: bool = False,
    root: Root = ROOT,
):
    result = analysis.list_records(
        output,
        root=root,
        kind=kind,
        method=method,
        dataset=dataset,
        seed=seed,
        status=status,
        tag=tag,
    )
    display.listing(result)
    save_result(result, save, overwrite, root)


@app.command()
def show(
    id: str,
    output: str = "experiments",
    folder: str = "runs",
    root: Root = ROOT,
    save: Path | None = None,
    overwrite: bool = False,
):
    result = analysis.detail(id, output, root=root, folder=folder)
    display.detail(result)
    save_result(result, save, overwrite, root)


@app.command()
def annotate(
    id: str,
    output: str = "experiments",
    folder: str = "runs",
    tags: Annotated[list[str] | None, typer.Option("--tag")] = None,
    note: str | None = None,
    name: str | None = None,
    clear_tags: bool = False,
    root: Root = ROOT,
):
    analysis.annotate(
        id,
        output=output,
        root=root,
        folder=folder,
        tags=[] if clear_tags else tags,
        note=note,
        name=name,
    )


@app.command()
def compare(
    ids: Annotated[list[str], typer.Argument()],
    metric: Annotated[str, typer.Option()],
    csv: str | None = None,
    overwrite: bool = False,
    save: Path | None = None,
    output: str = "experiments",
    root: Root = ROOT,
):
    if csv and save and path(csv, root) == path(save, root):
        raise ValueError("CSV and JSON destinations must be different")
    result = analysis.table(
        ids, metric, destination=csv, overwrite=overwrite, output=output, root=root
    )
    display.comparison(result, metric)
    if csv:
        display.message(f"Saved CSV: {path(csv, root)}")
    save_result(result, save, overwrite, root)


@app.command()
def summarize(
    ids: Annotated[list[str], typer.Argument()],
    metric: Annotated[str, typer.Option()],
    save: Path | None = None,
    output: str = "experiments",
    overwrite: bool = False,
    root: Root = ROOT,
):
    result = analysis.summary(ids, metric, output=output, root=root)
    display.summary(result, metric)
    save_result(result, save, overwrite, root)


@app.command()
def paired(
    left: Annotated[list[str], typer.Option("--left")],
    right: Annotated[list[str], typer.Option("--right")],
    metric: str,
    test: Annotated[str, typer.Option()],
    save: Path | None = None,
    output: str = "experiments",
    overwrite: bool = False,
    root: Root = ROOT,
):
    result = analysis.paired(left, right, metric, test=test, output=output, root=root)
    display.paired(result, metric)
    save_result(result, save, overwrite, root)


@app.command()
def plot(
    ids: Annotated[list[str], typer.Argument()],
    metric: Annotated[str, typer.Option()],
    destination: Annotated[str, typer.Option()],
    split: Annotated[str, typer.Option()],
    title: str | None = None,
    dpi: int = 300,
    overwrite: bool = False,
    output: str = "experiments",
    root: Root = ROOT,
):
    fig = analysis.curves(
        ids, metric, split=split, title=title, output=output, root=root
    )
    saved = analysis.save_figure(
        fig, destination, dpi=dpi, overwrite=overwrite, root=root
    )
    display.message(f"Saved figure: {saved}")


if __name__ == "__main__":
    app()
