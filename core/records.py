"""Local records, OS-released writer locks and plain PyTorch checkpoints."""

import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock, Timeout

from core.config import ROOT, Config, dumps, path
from core.platforms import Platforms


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value=None):
    value = value or uuid.uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("ID must contain only letters, digits, '_' and '-'")
    return value


def atomic(file: Path, value, *, torch_file=False):
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(file.name + ".tmp")
    try:
        if torch_file:
            torch.save(value, tmp)
        else:
            tmp.write_text(
                value
                if isinstance(value, str)
                else json.dumps(value, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        os.replace(tmp, file)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0], np.asarray(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def environment(root):
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            )
        )
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
        result["git_dirty"] = None
    return result


def directory(config: Config, kind: str, id: str, root=ROOT):
    return path(config.output, root) / kind / identifier(id)


@contextmanager
def locked(target: Path):
    """The lock lives outside the replaceable directory; never unlink its inode."""
    lock_dir = target.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_dir / f"{target.name}.lock"), timeout=0):
        yield


def snapshot(target: Path):
    meta = read_json(target / "record.json")
    if meta["status"] == "running" and meta["kind"] != "study":
        try:
            with locked(target):
                meta = read_json(target / "record.json")
                if meta["status"] == "running":
                    meta["status"] = "interrupted"
        except Timeout:
            pass
    if meta["kind"] == "study":
        import optuna

        from core.tuning import storage

        study = optuna.load_study(study_name=meta["id"], storage=storage(target))
        trials = study.trials
        meta["trial_counts"] = {
            state: sum(trial.state.name == state for trial in trials)
            for state in ("COMPLETE", "FAIL", "PRUNED")
        }
        meta["unfinished_trials"] = [
            trial.number for trial in trials if not trial.state.is_finished()
        ]
        if any(item["status"] == "running" for item in meta["launches"]):
            meta["status"] = "unconfirmed"
            meta["status_note"] = "有未结束记录，当前是否仍在运行未确认"
    return meta


def fresh(target: Path, *, overwrite=False, kind=None):
    if target.is_symlink():
        raise ValueError("Refusing symlink output directory")
    if target.exists():
        if not overwrite:
            raise FileExistsError(target)
        metadata = target / "record.json"
        if not metadata.is_file() or read_json(metadata).get("id") != target.name:
            raise ValueError("Refusing to overwrite a directory not owned by Research")
        if kind is not None and read_json(metadata).get("kind") != kind:
            raise ValueError("Cannot overwrite a different kind of record")
        shutil.rmtree(target)
    target.mkdir(parents=True)


def numbers(values):
    result = {key: float(value) for key, value in values.items()}
    if not result or any(not math.isfinite(v) for v in result.values()):
        raise ValueError("Metrics must be nonempty finite scalars")
    return result


def coordinate(epoch, step):
    if (epoch is None) == (step is None):
        raise ValueError(
            "Provide exactly one progress coordinate: epoch or custom step"
        )
    return {"epoch": epoch} if epoch is not None else {"step": step}


class Run:
    """Use under locked(directory). Custom loops call log/save_checkpoint explicitly."""

    def __init__(
        self,
        config: Config,
        target: Path,
        *,
        kind="train",
        resume=False,
        root=ROOT,
        source=None,
        invocation=None,
    ):
        self.config, self.directory = config, target
        self.platforms = None
        self.last = None
        if resume:
            self.meta = read_json(target / "record.json")
            self.last = read_json(target / "checkpoint.json")
        else:
            self.meta = {
                "id": target.name,
                "kind": kind,
                "name": config.name,
                "method": config.method,
                "dataset": config.data.name,
                "seed": config.seed,
                "tags": config.tags,
                "note": config.note,
                "protocol": config.protocol.model_dump(),
                "created": now(),
                "source": (config.provenance | (source or {})) or None,
                "launches": [],
            }
            atomic(target / "config.toml", dumps(config))
        self.launch = len(self.meta["launches"]) + 1
        start = {
            "number": self.launch,
            "started": now(),
            "status": "running",
            "invocation": invocation or [],
            "resume": {
                key: self.last[key]
                for key in ("launch", "epoch", "step")
                if key in self.last
            }
            if self.last
            else None,
            "environment": environment(root),
            "root": str(Path(root).resolve()),
        }
        self.meta["launches"].append(start)
        self.meta.update(status="running", updated=now())
        atomic(target / f"config-{self.launch}.toml", dumps(config))
        self.persist()
        self.text(f"launch {self.launch} started")

    def persist(self):
        atomic(self.directory / "record.json", self.meta)

    def open_platforms(self):
        self.platforms = Platforms(
            self.config,
            self.directory / f"launch-{self.launch}",
            f"{self.meta['id']}-{self.launch}",
        )

    def text(self, message):
        with (self.directory / "log.txt").open("a", encoding="utf-8") as file:
            file.write(f"{now()} [{self.launch}] {message}\n")

    def log(self, values, epoch=None, split="validation", *, step=None):
        values = numbers(values)
        row = {
            "launch": self.launch,
            **coordinate(epoch, step),
            "split": split,
            "metrics": values,
        }
        with (self.directory / f"metrics-{self.launch}.jsonl").open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        if self.platforms:
            self.platforms.log(
                {f"{split}/{key}": value for key, value in values.items()},
                epoch if epoch is not None else step,
            )

    def save_checkpoint(self, state, epoch=None, score=None, *, step=None):
        progress = coordinate(epoch, step)
        previous = self.last
        top = list(previous["top"]) if previous else []
        name = f"checkpoint-{self.launch}-{next(iter(progress.values()))}-{uuid.uuid4().hex}.pt"
        entry = {"file": name, **progress, "launch": self.launch}
        if score is not None:
            self.config.train.require("direction")
            top.append(entry | {"score": numbers({"score": score})["score"]})
            top.sort(
                key=lambda item: item["score"],
                reverse=self.config.train.direction == "max",
            )
            top = top[: self.config.train.keep_top_k]
        management = entry | {"top": top}
        # Never overwrite a referenced state, even if the same epoch is saved twice.
        atomic(self.directory / name, state, torch_file=True)
        atomic(self.directory / "checkpoint.json", management)
        self.last = management | {"state": state}
        if previous:
            old = {previous["file"]} | {item["file"] for item in previous["top"]}
            keep = {name} | {item["file"] for item in top}
            for obsolete in old - keep:
                (self.directory / obsolete).unlink()

    def finish(self, status, error=None, result=None, result_source=None):
        if self.platforms:
            self.platforms.close(status, error)
            self.platforms = None
        end = {"status": status, "ended": now(), "error": error}
        self.meta["launches"][-1].update(end)
        self.meta.update(end, updated=now())
        if status == "completed":
            if result_source is None:
                raise ValueError("Completed results require an explicit source")
            self.meta["result"] = numbers(result)
            self.meta["result_source"] = result_source
        self.persist()
        self.text(f"{status}: {error or ''}")


def load_state(file):
    """Default loader; methods may explicitly supply their own load_state(file)."""
    return torch.load(file, map_location="cpu", weights_only=True)


def checkpoint_info(target: Path, selection="best"):
    last = read_json(target / "checkpoint.json")
    if selection == "last":
        return {key: value for key, value in last.items() if key != "top"}
    if selection == "best":
        if not last["top"]:
            raise ValueError("Method has not saved ranked checkpoints")
        return last["top"][0]
    for item in last["top"]:
        if item["file"] == selection:
            return item
    raise ValueError("Select last, best or a checkpoint in the retained Top-K")


def checkpoint(target: Path, selection="best"):
    return target / checkpoint_info(target, selection)["file"]


def history(target: Path):
    """Keep only the committed lineage, retaining raw abandoned rows on disk."""
    rows = []
    for file in sorted(
        target.glob("metrics-*.jsonl"), key=lambda f: int(f.stem.split("-")[1])
    ):
        with file.open(encoding="utf-8") as stream:
            lines = stream.readlines()
        for i, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1 or line.endswith("\n"):
                    raise
                # A killed append is not a result. The next launch writes a new file.
    meta = read_json(target / "record.json")
    if not (target / "checkpoint.json").exists():
        return (
            rows
            if (
                meta["status"] == "completed"
                and meta.get("result_source", {}).get("kind")
                in {"custom_final", "test"}
            )
            else []
        )
    last = read_json(target / "checkpoint.json")
    axis = "epoch" if "epoch" in last else "step"
    bounds = {last["launch"]: last[axis]}
    launch = last["launch"]
    while meta["launches"][launch - 1]["resume"]:
        source = meta["launches"][launch - 1]["resume"]
        launch = source["launch"]
        bounds[launch] = source[axis]
    return [
        row
        for row in rows
        if row["launch"] in bounds and row[axis] <= bounds[row["launch"]]
    ]
