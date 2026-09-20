"""Human-readable views. Never modify results or write experiment records."""

from datetime import datetime, timezone
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
    groups = [data] if isinstance(data, dict) else data
    groups = [group for group in groups if group]
    for index, group in enumerate(groups):
        for row, (key, item) in enumerate(group.items()):
            view.add_row(
                Text(key),
                Text(value(item)),
                end_section=index < len(groups) - 1 and row == len(group) - 1,
            )
    console.print(view)


def listing(records):
    if not records:
        message("No matching records.")
        return
    keys = ("kind", "method", "dataset", "seed", "status")
    table(
        "Experiments",
        ("Experiment", "Type", "Method", "Dataset", "Seed", "Status"),
        (
            [
                f"{row['name']}\n{row['id']}" if row.get("name") else row["id"],
                *(row.get(key) for key in keys),
            ]
            for row in records
        ),
    )


def timestamp(text):
    if text is None:
        return None
    return (
        datetime.fromisoformat(text)
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def detail(record):
    launches = record.get("launches", [])
    info = {
        "Type / Status": f"{record['kind']} / {record['status']}",
        **{
            key.title(): record[key]
            for key in ("method", "dataset", "seed", "name", "tags", "note")
            if record.get(key) is not None and record[key] != "" and record[key] != []
        },
    }
    selection, protocol, times, notices, counts = {}, {}, {}, {}, {}
    for key, title in (("result_source", "Result selection"), ("source", "Source")):
        if record.get(key):
            selection[title] = record[key]
    for key, item in record.get("protocol", {}).items():
        protocol[f"Protocol.{key}"] = item
    started = launches[0].get("started") if launches else record.get("created")
    ended = record.get("ended") or (launches[-1].get("ended") if launches else None)
    if started:
        times["Started (UTC)"] = timestamp(started)
    if ended:
        times["Ended (UTC)"] = timestamp(ended)
    error = record.get("error")
    if len(launches) <= 1:
        error = error or (launches[0].get("error") if launches else None)
        if error:
            notices["Error"] = error
    elif error and error != launches[-1].get("error"):
        notices["Error"] = error
    if record.get("status_note"):
        notices["Notice"] = (
            "Unfinished launch records exist; current activity is unknown."
            if record["kind"] == "study" and record["status"] == "unconfirmed"
            else record["status_note"]
        )
    if "trials" in record:
        counts.update(record.get("trial_counts", {}))
        counts["Unfinished trials"] = record.get("unfinished_trials", [])
    fields(
        f"{record['kind'].title()}: {record['id']}",
        [info, selection, protocol, times, notices, counts],
    )
    if record.get("result"):
        table("Results", ("Metric", "Value"), record["result"].items())
    if len(launches) > 1:
        table(
            "Launch history",
            ("Launch", "Status", "Time (UTC)", "Resume source"),
            (
                (
                    r["number"],
                    r["status"],
                    f"Start {value(timestamp(r.get('started')))}\n"
                    f"End   {value(timestamp(r.get('ended')))}",
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
        [
            {
                "Test": result["test"],
                "Alternative": result["alternative"],
                "Pairs": result["pairs"],
            },
            {
                "Mean difference (left − right)": result["mean_difference"],
                "p-value": result["p_value"],
                "Status": result["status"],
                **({"Reason": result["reason"]} if "reason" in result else {}),
            },
        ],
    )
    table(
        "Descriptive statistics",
        ("Side", "Mean", "Sample std"),
        [
            (side.title(), result[side]["mean"], result[side]["std"])
            for side in ("left", "right")
        ],
    )


class RunDisplay:
    """Optional CLI feedback; custom methods need no display interface."""

    def __init__(self):
        self.progress = None

    def start(self, run, *, supervised=False):
        self.run = run
        self.started = monotonic()
        message(
            f"Starting {run.meta['kind']} · {run.meta['id']} · {run.config.method} · {run.config.device}"
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
        text = f"{record['status'].title()} {record['kind']} · {record['id']} · saved to {self.run.directory}"
        if record.get("error"):
            text += f" · {record['error']}"
        message(text)
        if record["status"] == "completed" and record.get("result"):
            table("Results", ("Metric", "Value"), record["result"].items())
