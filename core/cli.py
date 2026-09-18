"""Thin CLI over synchronous Python functions."""

import sys
from pathlib import Path
from typing import Annotated

import typer

from core import analysis, groups, runner, tuning
from core.config import ROOT, load

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
group_app = typer.Typer(no_args_is_help=True)
app.add_typer(group_app, name="group")
Root = Annotated[Path, typer.Option(help="Project root for relative paths")]
Overrides = Annotated[
    list[str] | None,
    typer.Option("--set", help='TOML override, e.g. params.lr=0.01 or device="cuda:0"'),
]


def config(file, overrides, root):
    return load(file, overrides or [], root)


@app.command()
def prepare(file: Path, root: Root = ROOT, set: Overrides = None):
    """Explicitly prepare data; existing prepared data is never silently rebuilt."""
    typer.echo(runner.prepare(config(file, set, root), root))


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
    typer.echo(
        runner.execute(
            config(file, set, root),
            id=id,
            resume=resume,
            overwrite=overwrite,
            root=root,
            invocation=sys.argv,
        )
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
    typer.echo(
        runner.test(
            config(file, set, root),
            source=source,
            selection=checkpoint,
            external=external,
            id=id,
            overwrite=overwrite,
            root=root,
            invocation=sys.argv,
        )
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
    typer.echo(
        tuning.export(
            load(file, root=root),
            study,
            destination,
            trial_number=trial,
            overwrite=overwrite,
            root=root,
        )
    )


@group_app.command("create")
def group_create(
    files: Annotated[list[Path], typer.Argument()],
    seeds: Annotated[list[int], typer.Option("--seed")],
    id: str | None = None,
    root: Root = ROOT,
):
    typer.echo(
        groups.create(
            [load(file, root=root) for file in files], seeds, id=id, root=root
        )
    )


@group_app.command("run")
def group_run(file: Path, id: str, action: str | None = None, root: Root = ROOT):
    """Continue serially; --action resume/rerun handles failed members explicitly."""
    typer.echo(groups.run_group(load(file, root=root), id, action=action, root=root))


@group_app.command("test")
def group_test(
    file: Path,
    id: str,
    members: Annotated[list[str], typer.Option("--member")],
    root: Root = ROOT,
):
    typer.echo(
        analysis.pretty(
            groups.test_members(load(file, root=root), id, members, root=root)
        )
    )


@app.command("list")
def list_runs(
    output: str = "experiments",
    kind: str | None = None,
    method: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    status: str | None = None,
    tag: str | None = None,
    root: Root = ROOT,
):
    typer.echo(
        analysis.pretty(
            analysis.list_records(
                output,
                root=root,
                kind=kind,
                method=method,
                dataset=dataset,
                seed=seed,
                status=status,
                tag=tag,
            )
        )
    )


@app.command()
def show(id: str, output: str = "experiments", folder: str = "runs", root: Root = ROOT):
    typer.echo(analysis.pretty(analysis.detail(id, output, root=root, folder=folder)))


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
    output: str = "experiments",
    root: Root = ROOT,
):
    typer.echo(
        analysis.pretty(
            analysis.table(
                ids,
                metric,
                destination=csv,
                overwrite=overwrite,
                output=output,
                root=root,
            )
        )
    )


@app.command()
def summarize(
    ids: Annotated[list[str], typer.Argument()],
    metric: Annotated[str, typer.Option()],
    output: str = "experiments",
    root: Root = ROOT,
):
    typer.echo(analysis.pretty(analysis.summary(ids, metric, output=output, root=root)))


@app.command()
def paired(
    left: Annotated[list[str], typer.Option("--left")],
    right: Annotated[list[str], typer.Option("--right")],
    metric: str,
    test: Annotated[str, typer.Option()],
    output: str = "experiments",
    root: Root = ROOT,
):
    typer.echo(
        analysis.pretty(
            analysis.paired(left, right, metric, test=test, output=output, root=root)
        )
    )


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
    typer.echo(
        analysis.save_figure(fig, destination, dpi=dpi, overwrite=overwrite, root=root)
    )


if __name__ == "__main__":
    app()
