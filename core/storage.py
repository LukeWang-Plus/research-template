"""Optional rclone transfers of independent datasets, runs and collection records.

One writer publishes increasing batch numbers by creating a unique marker last.
Readers pin that batch for the whole download. No directory rename or mutable
current pointer is assumed. Sources must be stable; readers and writers are
coordinated by the researcher, not by a distributed lock.
"""

import json
import re
import shutil
import subprocess
import tempfile
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from core.config import ROOT, path
from core.records import identifier, locked, read_json

KINDS = {"datasets", "runs", "studies", "groups"}
BATCH = re.compile(r"[0-9]{20}-[0-9a-f]{32}")
RUN_FILES = re.compile(
    r"config(?:-[0-9]+)?\.toml|record\.json|metrics-[0-9]+\.jsonl|checkpoint\.json|log\.txt"
)
STUDY_FILES = re.compile(r"config(?:-[0-9]+)?\.toml|record\.json|study\.sqlite3")


@dataclass
class TransferResult:
    kind: str
    id: str
    status: str = "not_attempted"
    error: str | None = None
    published: bool = False


def format_results(results):
    return "\n".join(
        f"{item.kind}/{item.id}: {item.status}"
        + (" (published; cleanup incomplete)" if item.published and item.error else "")
        + (f" — {item.error}" if item.error else "")
        for item in results
    )


class TransferError(RuntimeError):
    def __init__(self, results):
        self.results = results
        super().__init__(
            "Artifact transfer failed; local computation records are unchanged"
        )


class Rclone:
    def __init__(self, config):
        remote = config.storage.remote if config.storage else None
        if not remote or not re.fullmatch(r"[^:/\\]+:.+", remote):
            raise ValueError(
                "Set storage.remote to a configured rclone remote:project/path"
            )
        self.remote = remote.rstrip("/")

    def run(self, *args, input=None):
        try:
            result = subprocess.run(
                ["rclone", *map(str, args)],
                input=input,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Install rclone and configure storage.remote to transfer artifacts"
            ) from exc
        if result.returncode:
            raise RuntimeError(
                f"rclone {args[0]} exited {result.returncode}: {result.stderr.strip()}"
            )
        return result.stdout

    def location(self, kind, id):
        if kind not in KINDS:
            raise ValueError(f"Unknown storage kind: {kind}")
        if not isinstance(id, str) or not id:
            raise ValueError("A nonempty artifact ID is required")
        if kind != "datasets":
            identifier(id)
        # Dataset names are research identities, not necessarily filesystem IDs.
        key = quote(id, safe="").replace(".", "%2E")
        return f"{self.remote}/{kind}/{key}"

    def batches(self, location):
        return sorted(
            name
            for name in self.run(
                "lsf", f"{location}/published", "--files-only"
            ).splitlines()
            if BATCH.fullmatch(name)
        )

    def latest(self, location, batches=None):
        for batch in reversed(self.batches(location) if batches is None else batches):
            try:
                marker = json.loads(self.run("cat", f"{location}/published/{batch}"))
            except json.JSONDecodeError:
                continue  # An interrupted marker write is not a publication.
            if isinstance(marker, dict) and marker.get("complete") is True:
                return batch, marker
        return None

    def upload(self, kind, id, source, *, overwrite, result, members=()):
        if not source.is_dir():
            raise NotADirectoryError(source)
        location = self.location(kind, id)
        self.run("mkdir", f"{location}/published")
        previous = self.batches(location)
        if not overwrite and self.latest(location, previous) is not None:
            raise FileExistsError(
                f"Remote {kind}/{id} already exists; authorize overwrite"
            )
        self.run("mkdir", f"{location}/batches")
        old_batches = [
            name.rstrip("/")
            for name in self.run(
                "lsf", f"{location}/batches", "--dirs-only"
            ).splitlines()
            if BATCH.fullmatch(name.rstrip("/"))
        ]
        number = int(previous[-1].split("-")[0]) + 1 if previous else 1
        batch = f"{number:020d}-{uuid.uuid4().hex}"
        content = f"{location}/batches/{batch}/content"
        self.run("mkdir", content)
        self.run("copy", source, content, "--create-empty-src-dirs")
        # A failed publication request can still have landed. Never delete this
        # batch on error, and never clean old content unless rclone reports success.
        # Collection links name the existing runs selected in this batch, not
        # pending group members or trials that never created a run directory.
        marker = {"complete": True, "members": list(members)}
        self.run("rcat", f"{location}/published/{batch}", input=json.dumps(marker))
        result.published = True
        for old in previous:
            self.run("deletefile", f"{location}/published/{old}")
        for old in old_batches:
            self.run("purge", f"{location}/batches/{old}")

    def fetch(self, kind, id, target, *, analysis_only):
        location = self.location(kind, id)
        published = self.latest(location)
        if published is None:
            raise FileNotFoundError(f"No published remote {kind}/{id}")
        batch, marker = published
        content = f"{location}/batches/{batch}/content"
        members = [identifier(member) for member in marker["members"]]
        if not analysis_only:
            self.run("copy", content, target, "--create-empty-src-dirs")
            return members
        if kind == "datasets":
            raise ValueError("Datasets require a full download")
        names = self.run("lsf", content, "--files-only").splitlines()
        if kind == "groups":
            meta = json.loads(self.run("cat", f"{content}/record.json"))
            wanted = {"record.json"} | {
                f"{identifier(member)}.toml" for member in meta["members"]
            }
            selected = [name for name in names if name in wanted]
        else:
            pattern = RUN_FILES if kind == "runs" else STUDY_FILES
            selected = [name for name in names if pattern.fullmatch(name)]
        if "record.json" not in selected:
            raise ValueError(f"Remote {kind}/{id} has no record.json")
        self.run(
            "copy",
            content,
            target,
            "--files-from-raw",
            "-",
            input="\n".join(selected) + "\n",
        )
        return members


def _members(kind, id, target):
    if kind == "groups":
        return [
            identifier(member)
            for member in read_json(target / "record.json")["members"]
        ]
    if kind == "studies":
        import optuna

        from core.tuning import storage

        backend = storage(target)
        try:
            study = optuna.load_study(study_name=id, storage=backend)
            return [
                identifier(trial.user_attrs["run_id"])
                for trial in study.trials
                if "run_id" in trial.user_attrs and "config" in trial.user_attrs
            ]
        finally:
            backend.remove_session()
            backend.engine.dispose()
    return []


def _local(config, kind, id, root, explicit=None):
    if kind not in KINDS:
        raise ValueError(f"Unknown storage kind: {kind}")
    if kind == "datasets":
        if explicit is None:
            raise ValueError(
                "Dataset transfers require an explicit source/destination directory"
            )
        return path(explicit, root)
    if explicit is not None:
        raise ValueError("Only dataset transfers accept an explicit directory")
    if not isinstance(id, str) or not id:
        raise ValueError("An explicit run/study/group ID is required")
    return path(config.output, root) / kind / identifier(id)


def _batch(objects, operation):
    results = [TransferResult(kind, id) for kind, id, _ in objects]
    for (kind, id, target), result in zip(objects, results, strict=True):
        try:
            operation(kind, id, target, result)
            result.status = "success"
        except Exception as exc:
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            break
    return results


def upload(
    config,
    kind,
    id=None,
    *,
    source=None,
    overwrite=False,
    root=ROOT,
    _overwrite_runs=None,
):
    """Upload an object, including existing associated runs for a study/group.

    Return per-object results, stopping after the first failure. Explicit calls
    work even when storage.enabled is false. Dataset IDs default to data.name.
    """
    id = config.data.name if kind == "datasets" and id is None else id
    target = _local(config, kind, id, root, source)
    objects = [(kind, id, target)]
    objects += [
        ("runs", member, local)
        for member in _members(kind, id, target)
        if (local := _local(config, "runs", member, root)).is_dir()
    ]
    return _upload(
        config, objects, overwrite=overwrite, _overwrite_runs=_overwrite_runs
    )


def _upload(config, objects, *, overwrite, _overwrite_runs=None):
    members = [id for kind, id, _ in objects if kind == "runs"]

    def transfer(kind, id, target, result):
        Rclone(config).upload(
            kind,
            id,
            target,
            overwrite=(id in _overwrite_runs)
            if kind == "runs" and _overwrite_runs is not None
            else overwrite,
            result=result,
            members=members if kind in {"studies", "groups"} else (),
        )

    return _batch(objects, transfer)


def _install(stage, target, *, analysis_only):
    if analysis_only and target.exists():
        # Only known analysis files were fetched. Remove stale analysis records,
        # never weights, predictions or method-specific files.
        if (stage / "record.json").is_file():
            kind = read_json(stage / "record.json")["kind"]
            if kind == "group":
                old = (
                    read_json(target / "record.json")
                    if (target / "record.json").exists()
                    else {}
                )
                names = {"record.json"} | {
                    f"{identifier(m)}.toml" for m in old.get("members", [])
                }
            else:
                pattern = STUDY_FILES if kind == "study" else RUN_FILES
                names = {
                    file.name
                    for file in target.iterdir()
                    if pattern.fullmatch(file.name)
                }
            for name in names - {file.name for file in stage.iterdir()}:
                (target / name).unlink(missing_ok=True)
        for file in stage.iterdir():
            file.replace(target / file.name)
        return
    backup = target.with_name(f".{target.name}-old-{uuid.uuid4().hex}")
    if target.exists():
        target.rename(backup)
    try:
        stage.rename(target)
    except BaseException:
        if backup.exists():
            backup.rename(target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def download(
    config,
    kind,
    id=None,
    *,
    destination=None,
    analysis_only=True,
    overwrite=False,
    root=ROOT,
):
    """Download analysis records (default) or complete objects to local layout.

    Existing targets require overwrite. Each full object is downloaded to a
    sibling temporary directory before replacing its target under the writer lock.
    """
    id = config.data.name if kind == "datasets" and id is None else id
    target = _local(config, kind, id, root, destination)
    members = []

    def transfer(kind, id, target, result):
        target.parent.mkdir(parents=True, exist_ok=True)
        # Data preparation has no shared lock contract. Do not invent one here.
        with locked(target) if kind != "datasets" else nullcontext():
            if target.is_symlink():
                raise ValueError("Refusing symlink download directory")
            if target.exists() and not overwrite:
                raise FileExistsError(target)
            if target.exists() and not target.is_dir():
                raise NotADirectoryError(target)
            with tempfile.TemporaryDirectory(
                prefix=f".{target.name}-download-", dir=target.parent
            ) as temporary:
                stage = Path(temporary) / "content"
                stage.mkdir()
                associated = Rclone(config).fetch(
                    kind, id, stage, analysis_only=analysis_only
                )
                _install(stage, target, analysis_only=analysis_only)
                members.extend(associated)

    results = _batch([(kind, id, target)], transfer)
    if results[0].status == "success" and members:
        objects = [
            ("runs", member, _local(config, "runs", member, root)) for member in members
        ]
        results.extend(_batch(objects, transfer))
    return results


def enabled(config):
    return config.storage is not None and config.storage.enabled


def _automatic(results):
    print(format_results(results))
    if any(result.status != "success" for result in results):
        raise TransferError(results)


def auto_upload(config, kind, id=None, **kwargs):
    if enabled(config):
        _automatic(upload(config, kind, id, **kwargs))


def auto_upload_runs(config, ids, *, root=ROOT):
    if enabled(config):
        objects = [("runs", id, _local(config, "runs", id, root)) for id in ids]
        _automatic(_upload(config, objects, overwrite=False))
