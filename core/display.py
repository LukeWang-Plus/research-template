"""Human-readable views. Never modify results or write experiment records."""

from time import monotonic

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

console = Console()


def value(item):
    if item is None:
        return "—"
    if isinstance(item, float):
        return format(item, ".6g")
    if isinstance(item, dict):
        return "\n".join(f"{key}={value(val)}" for key, val in item.items()) or "—"
    if isinstance(item, (list, tuple)):
        return ", ".join(value(val) for val in item) or "—"
    return str(item)


def message(text, *, style=None):
    console.print(Text(str(text), style=style or ""))


def table(title, columns, rows):
    view = Table(title=Text(title), header_style="bold", show_lines=False)
    for column in columns:
        view.add_column(column, overflow="fold")
    for row in rows:
        view.add_row(*(Text(value(cell)) for cell in row))
    console.print(view)


def fields(title, data):
    view = Table(title=Text(title), header_style="bold")
    view.add_column("Field", style="bold", overflow="fold")
    view.add_column("Value", overflow="fold")
    for key, item in data.items():
        view.add_row(Text(key), Text(value(item)))
    console.print(view)


def listing(records):
    if not records:
        message("No matching records.")
        return
    keys = ("id", "kind", "method", "dataset", "seed", "status", "name")
    table(
        "Experiments",
        ("ID", "Type", "Method", "Dataset", "Seed", "Status", "Name"),
        ([row.get(key) for key in keys] for row in records),
    )


def detail(record):
    fields(
        f"{record['kind'].title()}: {record['id']}",
        {
            key.title(): record[key]
            for key in (
                "kind",
                "status",
                "name",
                "method",
                "dataset",
                "seed",
                "tags",
                "note",
                "created",
                "ended",
            )
            if record.get(key) is not None and record[key] != "" and record[key] != []
        },
    )
    if record.get("status_note"):
        notice = (
            "Unfinished launch records exist; current activity is unknown."
            if record["kind"] == "study" and record["status"] == "unconfirmed"
            else record["status_note"]
        )
        message(notice, style="yellow")
    if record.get("result"):
        table("Results", ("Metric", "Value"), record["result"].items())
    for key, title in (
        ("result_source", "Result selection"),
        ("source", "Source"),
        ("protocol", "Protocol"),
    ):
        if record.get(key):
            fields(title, record[key])
    if record.get("error"):
        fields("Error", {"Reason": record["error"]})
    launches = record.get("launches", [])
    if launches:
        table(
            "Launch history",
            ("Launch", "Status", "Started", "Ended", "Resume source"),
            (
                (
                    r["number"],
                    r["status"],
                    r.get("started"),
                    r.get("ended"),
                    r.get("resume"),
                )
                for r in launches
            ),
        )
        errors = {
            f"Launch {r['number']}": r["error"] for r in launches if r.get("error")
        }
        if errors:
            fields("Launch errors", errors)
    if "trials" in record:
        fields(
            "Trials overview",
            record.get("trial_counts", {})
            | {"Unfinished": record.get("unfinished_trials", [])},
        )
        table(
            "Trials",
            ("Trial", "Status", "Value", "Parameters", "Run ID", "Retry of"),
            (
                (
                    t["number"],
                    t["status"],
                    t["value"],
                    t["params"],
                    t["run_id"],
                    t["retry_of"],
                )
                for t in record["trials"]
            ),
        )
        errors = {
            f"Trial {t['number']}": t["error"]
            for t in record["trials"]
            if t.get("error")
        }
        if errors:
            fields("Trial errors", errors)
    if "member_states" in record:
        table("Members", ("Run ID", "Status"), record["member_states"].items())


def flatten(data, prefix=""):
    result = {}
    for key, item in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def comparison(rows, metric):
    warnings = list(dict.fromkeys(row["warning"] for row in rows if row.get("warning")))
    for warning in warnings:
        message(f"WARNING: {warning}", style="bold yellow")
    table(
        f"Comparison · {metric}",
        ("ID", "Method", "Dataset", "Seed", "Value"),
        ((r["id"], r["method"], r["dataset"], r["seed"], r["value"]) for r in rows),
    )
    for key, title in (
        ("setup", "Training differences"),
        ("protocol", "Protocol differences"),
    ):
        setups = [flatten(row[key]) for row in rows]
        names = sorted(set().union(*(item.keys() for item in setups)))
        different = [
            name
            for name in names
            if any(
                (name in item, item.get(name))
                != (name in setups[0], setups[0].get(name))
                for item in setups[1:]
            )
        ]
        if different:
            table(
                title,
                ("Setting", *(r["id"] for r in rows)),
                (
                    (name, *(item.get(name, "<absent>") for item in setups))
                    for name in different
                ),
            )


def summary(result, metric):
    table(
        f"Summary · {metric}",
        ("Runs", "Mean", "Sample std"),
        [(result["n"], result["mean"], result["std"])],
    )


def paired(result, metric):
    fields(
        f"Paired comparison · {metric}",
        {
            "Test": result["test"],
            "Alternative": result["alternative"],
            "Pairs": result["pairs"],
        },
    )
    table(
        "Descriptive statistics",
        ("Side", "Mean", "Sample std"),
        [
            (side.title(), result[side]["mean"], result[side]["std"])
            for side in ("left", "right")
        ],
    )
    fields(
        "Test result",
        {
            "Mean difference (left − right)": result["mean_difference"],
            "p-value": result["p_value"],
            "Status": result["status"],
            **({"Reason": result["reason"]} if "reason" in result else {}),
        },
    )


class RunDisplay:
    """Optional CLI feedback; custom methods need no display interface."""

    def __init__(self):
        self.progress = None

    def start(self, run, *, supervised=False):
        self.run = run
        self.started = monotonic()
        fields(
            f"Starting {run.meta['kind']}: {run.meta['id']}",
            {
                "Method": run.config.method,
                "Device": run.config.device,
                "Output": run.directory,
                **(
                    {"Checkpoint": run.meta["source"]["checkpoint"]}
                    if (run.meta.get("source") or {}).get("checkpoint")
                    else {}
                ),
            },
        )
        if supervised:
            self.total = run.config.train.epochs
            self.metric = run.config.train.metric
            if console.is_terminal:
                self.progress = Progress(
                    TextColumn("Epoch {task.completed}/{task.total}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    TextColumn("{task.fields[metrics]}"),
                    console=console,
                )
                self.task = self.progress.add_task(
                    "Training",
                    total=self.total,
                    completed=run.last["epoch"] if run.last else 0,
                    metrics="",
                )
                self.progress.start()

    def epoch(self, epoch, loss, score):
        metrics = f"train/loss={value(loss)}  validation/{self.metric}={value(score)}"
        if self.progress:
            self.progress.update(self.task, completed=epoch, metrics=metrics)
        else:
            message(
                f"Epoch {epoch}/{self.total}  elapsed={monotonic() - self.started:.1f}s  {metrics}"
            )

    def finish(self):
        if self.progress:
            self.progress.stop()
            self.progress = None
        record = self.run.meta
        fields(
            f"Finished {record['kind']}: {record['id']}",
            {
                "Status": record["status"],
                "Output": self.run.directory,
                **({"Error": record["error"]} if record.get("error") else {}),
            },
        )
        if record["status"] == "completed" and record.get("result"):
            table("Results", ("Metric", "Value"), record["result"].items())
