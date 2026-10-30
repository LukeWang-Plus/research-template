"""Serial experiment groups; independent members can also run from any terminal."""

from core import storage
from core.config import ROOT, Config, dumps, load
from core.records import atomic, directory, fresh, identifier, locked, now, read_json
from core.runner import execute, test


def create(configs: list[Config], seeds: list[int], *, id=None, root=ROOT):
    if not configs or not seeds:
        raise ValueError("Provide at least one configuration and seed")
    if len({config.output for config in configs}) != 1:
        raise ValueError("All members must use the same output root")
    id = identifier(id)
    target = directory(configs[0], "groups", id, root)
    with locked(target):
        fresh(target)
        members = []
        for config in configs:
            for seed in seeds:
                actual = Config.model_validate(
                    config.model_dump()
                    | {"seed": seed, "provenance": config.provenance | {"group": id}}
                )
                member = identifier()
                atomic(target / f"{member}.toml", dumps(actual))
                members.append(member)
        atomic(
            target / "record.json",
            {
                "id": id,
                "kind": "group",
                "name": None,
                "tags": [],
                "note": "",
                "created": now(),
                "status": "pending",
                "members": members,
            },
        )
    return id


def run_group(config, id, *, action=None, root=ROOT, display=None):
    if action not in (None, "resume", "rerun"):
        raise ValueError("Failed members require resume or rerun")
    target = directory(config, "groups", id, root)
    overwritten = set()
    with locked(target):
        meta = read_json(target / "record.json")
        meta["status"] = "running"
        atomic(target / "record.json", meta)
        try:
            for member in meta["members"]:
                actual = load(target / f"{member}.toml", root=root)
                record = directory(actual, "runs", member, root) / "record.json"
                exists = record.parent.exists()
                if record.exists() and read_json(record)["status"] == "completed":
                    continue
                if exists and action is None:
                    raise ValueError(
                        f"Member {member} failed/interrupted: explicitly choose resume or rerun"
                    )
                execute(
                    actual,
                    id=member,
                    resume=exists and action == "resume",
                    overwrite=exists and action == "rerun",
                    root=root,
                    source={"group": id},
                    display=display,
                    _upload=False,
                )
                if exists:
                    overwritten.add(member)
            meta["status"] = "completed"
            meta.pop("error", None)
        except BaseException as exc:
            meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            atomic(target / "record.json", meta)
    storage.auto_upload(
        config,
        "groups",
        id,
        overwrite=action is not None,
        _overwrite_runs=overwritten,
        root=root,
    )
    return id


def test_members(config, id, members, *, root=ROOT, display=None):
    target = directory(config, "groups", id, root)
    meta = read_json(target / "record.json")
    if not members or not set(members) <= set(meta["members"]):
        raise ValueError("Explicitly select existing group member IDs")
    results = []
    for member in members:
        actual = load(target / f"{member}.toml", root=root)
        results.append(
            test(actual, source=member, root=root, display=display, _upload=False)
        )
    storage.auto_upload_runs(config, results, root=root)
    return results
