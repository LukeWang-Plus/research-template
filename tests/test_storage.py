"""Storage contracts with isolated rclone local remotes and focused fault injection."""

import shutil
from types import SimpleNamespace

import optuna
import pytest
from typer.testing import CliRunner

from core import analysis, groups, plotting, runner, storage, tuning
from core.cli import app
from core.config import Config, Storage, dumps, load
from core.records import atomic, checkpoint, history, locked, read_json
from tests.test_workflow import TrialFailure, cfg


@pytest.fixture
def remote(tmp_path, monkeypatch):
    if shutil.which("rclone") is None:
        pytest.skip("rclone is not installed")
    # An empty isolated config and environment-defined local backend never read
    # the researcher's rclone configuration or credentials.
    config_file = tmp_path / "rclone.conf"
    config_file.touch()
    monkeypatch.setenv("RCLONE_CONFIG", str(config_file))
    monkeypatch.setenv("RCLONE_CONFIG_ARTIFACT_TEST_TYPE", "local")
    return Storage(remote=f"artifact_test:{(tmp_path / 'remote').as_posix()}")


def simple_method():
    def run(config, run, root, trial):
        run.log({"accuracy": 0.5}, 1)
        run.save_checkpoint({"seed": config.seed}, 1, 0.5)
        return {"accuracy": 0.5}

    return SimpleNamespace(
        run=run,
        test=lambda config, run, root, file, state: {"accuracy": 0.5},
        supports_resume=True,
        objective=lambda config, target, trial: 0.5,
    )


def local_run(config, root, id):
    target = root / config.output / "runs" / id
    target.mkdir(parents=True)
    atomic(target / "record.json", {"kind": "train", "id": id})
    atomic(target / "config.toml", dumps(config))
    (target / "checkpoint-old.pt").write_text("old weights")
    return target


def published(config, kind, id):
    client = storage.Rclone(config)
    location = client.location(kind, id)
    return client, location, client.batches(location)


def test_optional_storage_and_prepare_contract(tmp_path, monkeypatch):
    method = SimpleNamespace(prepare=lambda config, root: None, run=simple_method().run)
    monkeypatch.setattr(runner, "get", lambda name: method)
    monkeypatch.setattr(
        storage.Rclone, "run", lambda *a, **k: pytest.fail("rclone was called")
    )
    c = cfg(tmp_path)
    assert runner.prepare(c, tmp_path) is None
    c.storage = Storage(enabled=False)
    assert runner.prepare(c, tmp_path) is None
    runner.execute(c, id="offline", root=tmp_path)
    saved = load(tmp_path / "experiments/runs/offline/config.toml")
    assert saved.storage.enabled is False
    c.storage = Storage(remote="archive:project")
    with pytest.raises(ValueError, match="prepare.*return"):
        runner.prepare(c, tmp_path)
    c.storage = Storage(enabled=False)
    result = storage.upload(c, "runs", "offline", root=tmp_path)
    assert result[0].status == "failed" and "storage.remote" in result[0].error


def test_local_rclone_workflow_and_analysis(tmp_path, remote):
    c = cfg(tmp_path, storage=remote.model_dump())
    c.train.epochs = 1
    data = runner.prepare(c, tmp_path)
    client, location, batches = published(c, "datasets", c.data.name)
    assert len(batches) == 1
    assert "train.pt" in client.run("lsf", f"{location}/batches/{batches[0]}/content")
    runner.execute(c, id="one", root=tmp_path)
    target = tmp_path / "experiments/runs/one"
    runner.test(c, source="one", id="evaluation", root=tmp_path)
    assert (tmp_path / "experiments/runs/evaluation/predictions.pt").exists()
    c.storage = remote.model_copy(update={"enabled": False})
    assert storage.upload(c, "runs", "one", root=tmp_path)[0].status == "failed"
    download_root = tmp_path / "analysis"
    results = storage.download(c, "runs", "one", root=download_root)
    assert results[0].status == "success", storage.format_results(results)
    restored = download_root / "experiments/runs/one"
    assert not list(restored.glob("*.pt"))
    assert history(restored) == history(target)
    assert analysis.table(["one"], "accuracy", root=download_root)[0]["value"] >= 0
    figure = plotting.curves(
        ["one"], "accuracy", split="validation", root=download_root
    )
    assert len(figure.axes[0].lines) == 1
    import matplotlib.pyplot as plt

    plt.close(figure)
    assert storage.download(c, "runs", "one", root=download_root)[0].status == "failed"
    (restored / "local.pt").write_text("keep on analysis overwrite")
    (restored / "custom.json").write_text("custom analysis is not template analysis")
    (restored / "metrics-99.jsonl").write_text("obsolete")
    assert (
        storage.download(c, "runs", "one", overwrite=True, root=download_root)[0].status
        == "success"
    )
    assert (restored / "local.pt").exists() and (restored / "custom.json").exists()
    assert not (restored / "metrics-99.jsonl").exists()
    assert (
        storage.download(
            c, "runs", "one", analysis_only=False, overwrite=True, root=download_root
        )[0].status
        == "success"
    )
    assert checkpoint(restored).exists() and not (restored / "local.pt").exists()
    assert not (restored / "custom.json").exists()
    destination = tmp_path / "downloaded-data"
    assert (
        storage.download(c, "datasets", destination=destination, analysis_only=False)[
            0
        ].status
        == "success"
    )
    assert set(p.name for p in data.iterdir()) == set(
        p.name for p in destination.iterdir()
    )

    # Resume and source testing accept changed, non-research storage settings.
    c.train.epochs = 2
    c.storage = remote
    old_weights = {p.name for p in target.glob("*.pt")}
    runner.execute(c, id="one", resume=True, root=tmp_path)
    client, location, batches = published(c, "runs", "one")
    assert len(batches) == 1
    names = client.run("lsf", f"{location}/batches/{batches[0]}/content").splitlines()
    assert set(names) == {p.name for p in target.iterdir()}
    assert not (old_weights - {p.name for p in target.glob("*.pt")}) & set(names)
    c.storage = remote.model_copy(update={"enabled": False})
    runner.test(c, source="one", id="different-storage", root=tmp_path)
    assert (
        read_json(tmp_path / "experiments/runs/different-storage/record.json")["status"]
        == "completed"
    )


def test_publication_failures_and_cleanup(tmp_path, remote, monkeypatch):
    c = cfg(tmp_path, storage=remote.model_dump())
    target = local_run(c, tmp_path, "one")
    assert storage.upload(c, "runs", "one", root=tmp_path)[0].status == "success"
    client, location, old = published(c, "runs", "one")
    (target / "checkpoint-old.pt").unlink()
    (target / "checkpoint-new.pt").write_text("new weights")
    native = storage.Rclone.run

    def failed_copy(self, *args, **kwargs):
        if args[0] == "copy":
            raise RuntimeError("injected content transfer failure")
        return native(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage.Rclone, "run", failed_copy)
        result = storage.upload(c, "runs", "one", overwrite=True, root=tmp_path)[0]
    assert result.status == "failed" and not result.published
    assert client.batches(location) == old
    assert "checkpoint-old.pt" in client.run(
        "lsf", f"{location}/batches/{old[0]}/content"
    )
    assert (
        storage.download(
            c, "runs", "one", analysis_only=False, root=tmp_path / "old-reader"
        )[0].status
        == "success"
    )
    assert (tmp_path / "old-reader/experiments/runs/one/checkpoint-old.pt").exists()
    # A partial completion marker is not a published batch, even if listed.
    partial = "00000000000000000002-" + "f" * 32
    client.run("rcat", f"{location}/published/{partial}", input='{"complete":')
    assert client.latest(location)[0] == old[0]

    def uncertain_publish(self, *args, **kwargs):
        result = native(self, *args, **kwargs)
        if args[0] == "rcat":
            raise RuntimeError("publication response lost after marker landed")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(storage.Rclone, "run", uncertain_publish)
        result = storage.upload(c, "runs", "one", overwrite=True, root=tmp_path)[0]
    assert result.status == "failed"
    assert len(client.batches(location)) == 3
    assert client.run("lsf", f"{location}/batches/{old[0]}/content")
    assert (
        storage.download(
            c, "runs", "one", analysis_only=False, root=tmp_path / "new-reader"
        )[0].status
        == "success"
    )
    assert (tmp_path / "new-reader/experiments/runs/one/checkpoint-new.pt").exists()
    assert (
        storage.upload(c, "runs", "one", overwrite=True, root=tmp_path)[0].status
        == "success"
    )
    assert len(client.batches(location)) == 1
    assert (
        len(client.run("lsf", f"{location}/batches", "--dirs-only").splitlines()) == 1
    )

    def failed_cleanup(self, *args, **kwargs):
        if args[0] == "deletefile":
            raise RuntimeError("injected cleanup failure")
        return native(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage.Rclone, "run", failed_cleanup)
        result = storage.upload(c, "runs", "one", overwrite=True, root=tmp_path)[0]
    assert result.status == "failed" and result.published
    assert "cleanup incomplete" in storage.format_results([result])


def test_download_failure_and_writer_lock(tmp_path, remote, monkeypatch):
    c = cfg(tmp_path, storage=remote.model_dump())
    source = local_run(c, tmp_path, "one")
    assert storage.upload(c, "runs", "one", root=tmp_path)[0].status == "success"
    old = (source / "checkpoint-old.pt").read_bytes()

    def fail(self, kind, id, target, **kwargs):
        (target / "partial.pt").write_text("partial")
        raise RuntimeError("download failed halfway")

    with monkeypatch.context() as patch:
        patch.setattr(storage.Rclone, "fetch", fail)
        result = storage.download(
            c, "runs", "one", analysis_only=False, overwrite=True, root=tmp_path
        )[0]
    assert result.status == "failed"
    assert (source / "checkpoint-old.pt").read_bytes() == old
    assert not (source / "partial.pt").exists()
    with locked(source):
        result = storage.download(c, "runs", "one", overwrite=True, root=tmp_path)[0]
    assert result.status == "failed" and "Timeout" in result.error
    assert not list(source.parent.glob(".one-download-*"))


def test_batch_partial_success_and_group_download(tmp_path, remote, monkeypatch):
    c = cfg(tmp_path, storage=remote.model_dump())
    group = tmp_path / "experiments/groups/set"
    atomic(group / "record.json", {"kind": "group", "members": ["a", "b", "c"]})
    for id in ("a", "b", "c"):
        local_run(c, tmp_path, id)
        atomic(group / f"{id}.toml", dumps(c))
    meta = read_json(group / "record.json")
    meta["members"].append("pending")
    atomic(group / "record.json", meta)
    atomic(group / "pending.toml", dumps(c))
    (group / "custom.toml").write_text("not a member config")
    native = storage.Rclone.upload

    def fail_b(self, kind, id, source, **kwargs):
        if id == "b":
            raise RuntimeError("injected member failure")
        return native(self, kind, id, source, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage.Rclone, "upload", fail_b)
        results = storage.upload(c, "groups", "set", root=tmp_path)
    assert [r.id for r in results] == ["set", "a", "b", "c"]
    assert [r.status for r in results] == [
        "success",
        "success",
        "failed",
        "not_attempted",
    ]
    assert "injected" in results[2].error
    # Retry only the missing independent members; no group overwrite is needed.
    for id in ("b", "c"):
        assert storage.upload(c, "runs", id, root=tmp_path)[0].status == "success"
    root = tmp_path / "reader"
    results = storage.download(c, "groups", "set", root=root)
    assert all(r.status == "success" for r in results), storage.format_results(results)
    assert not (root / "experiments/groups/set/custom.toml").exists()
    assert set(p.name for p in (root / "experiments/groups/set").iterdir()) == {
        "record.json",
        "a.toml",
        "b.toml",
        "c.toml",
        "pending.toml",
    }
    assert all(
        not list((root / "experiments/runs" / id).glob("*.pt"))
        for id in ("a", "b", "c")
    )


def test_tune_batches_trial_states_and_continuation(tmp_path, monkeypatch, remote):
    c = cfg(tmp_path, storage=remote.model_dump())
    c.search.catch = ["tests.test_workflow.TrialFailure"]
    method = simple_method()
    normal = method.run

    def trials(config, run, root, trial):
        if trial.number == 0:
            raise TrialFailure("tolerated")
        if trial.number == 1:
            raise optuna.TrialPruned("pruned")
        return normal(config, run, root, trial)

    method.run = trials
    monkeypatch.setattr(runner, "get", lambda name: method)
    monkeypatch.setattr(tuning, "get", lambda name: method)
    tuning.tune(c, trials=3, id="search", root=tmp_path)
    root = tmp_path / "reader"
    results = storage.download(c, "studies", "search", root=root)
    assert all(r.status == "success" for r in results), storage.format_results(results)
    detail = analysis.detail("search", folder="studies", root=root)
    assert [t["status"] for t in detail["trials"]] == ["FAIL", "PRUNED", "COMPLETE"]
    ids = [t["run_id"] for t in detail["trials"]]
    assert [
        read_json(root / "experiments/runs" / id / "record.json")["status"]
        for id in ids
    ] == ["failed", "pruned", "completed"]
    c.storage = remote.model_copy(update={"enabled": False})
    tuning.tune(c, trials=1, id="search", continuation=True, root=tmp_path)
    # Continuation can change storage without changing research configuration;
    # the initial and launch snapshots keep their original settings.
    study_dir = tmp_path / "experiments/studies/search"
    assert load(study_dir / "config-2.toml").storage.enabled is False
    c.storage = remote
    tuning.tune(c, trials=1, id="search", continuation=True, root=tmp_path)
    assert (
        storage.download(c, "studies", "search", overwrite=True, root=root)[0].status
        == "success"
    )
    assert len(analysis.detail("search", folder="studies", root=root)["trials"]) == 5


def test_group_top_level_storage_and_failure_boundaries(tmp_path, monkeypatch):
    c = cfg(tmp_path, storage={"remote": "member:project"})
    method = simple_method()
    monkeypatch.setattr(runner, "get", lambda name: method)
    calls = []

    def capture(config, objects, *, overwrite, _overwrite_runs=None):
        calls.append((config.storage.remote, objects, overwrite, _overwrite_runs))
        assert all(
            read_json(target / "record.json")["status"] == "completed"
            for _, _, target in objects
        )
        return [storage.TransferResult(kind, id, "success") for kind, id, _ in objects]

    monkeypatch.setattr(storage, "_upload", capture)
    id = groups.create([c], [1, 2], id="set", root=tmp_path)
    top = Config.model_validate(c.model_dump() | {"storage": {"remote": "top:project"}})
    groups.run_group(top, id, root=tmp_path)
    assert len(calls) == 1 and calls[0][0] == "top:project" and calls[0][2] is False
    assert calls[0][3] == set()
    assert [kind for kind, _, _ in calls[0][1]] == ["groups", "runs", "runs"]
    members = read_json(tmp_path / "experiments/groups/set/record.json")["members"]
    for member in members:
        assert (
            load(tmp_path / "experiments/runs" / member / "config.toml").storage.remote
            == "member:project"
        )
    top.storage.enabled = False
    groups.test_members(top, id, members, root=tmp_path)
    assert len(calls) == 1
    top.storage.enabled = True
    tests = groups.test_members(top, id, members, root=tmp_path)
    assert len(calls) == 2 and [item[1] for item in calls[1][1]] == tests
    assert calls[1][0] == "top:project"

    def failed(config, run, root, trial):
        raise KeyboardInterrupt("interrupted")

    method.run = failed
    with pytest.raises(KeyboardInterrupt):
        runner.execute(top, id="failed", root=tmp_path)
    assert len(calls) == 2
    with pytest.raises(KeyboardInterrupt):
        groups.run_group(
            top,
            groups.create([c], [3], id="failed-group", root=tmp_path),
            root=tmp_path,
        )
    assert len(calls) == 2


@pytest.mark.parametrize("action", ["resume", "rerun"])
def test_group_member_overwrite_authorization(tmp_path, monkeypatch, remote, action):
    c = cfg(tmp_path, storage=remote.model_dump())
    method = simple_method()
    normal = method.run

    def run(config, run, root, trial):
        (run.directory / "payload.txt").write_text(f"local-{config.seed}")
        return normal(config, run, root, trial)

    method.run = run
    monkeypatch.setattr(runner, "get", lambda name: method)
    group = groups.create([c], [1, 2, 3], id="set", root=tmp_path)
    group_dir = tmp_path / "experiments/groups" / group
    recovering, new, skipped = read_json(group_dir / "record.json")["members"]
    old_batches = {}
    for member in (recovering, skipped):
        actual = load(group_dir / f"{member}.toml")
        runner.execute(actual, id=member, root=tmp_path, _upload=False)
        target = tmp_path / "experiments/runs" / member
        if member == recovering:
            meta = read_json(target / "record.json")
            meta["status"] = meta["launches"][-1]["status"] = "failed"
            atomic(target / "record.json", meta)
        (target / "payload.txt").write_text("remote-old")
        result = storage.upload(c, "runs", member, root=tmp_path)
        assert result[0].status == "success", storage.format_results(result)
        old_batches[member] = published(c, "runs", member)[2]
        (target / "payload.txt").write_text("local-unpublished")

    attempts = []
    native = storage.Rclone.upload

    def capture(self, kind, id, source, *, overwrite, **kwargs):
        attempts.append((kind, id, overwrite))
        return native(self, kind, id, source, overwrite=overwrite, **kwargs)

    monkeypatch.setattr(storage.Rclone, "upload", capture)
    with pytest.raises(storage.TransferError) as error:
        groups.run_group(c, group, action=action, root=tmp_path)
    assert attempts == [
        ("groups", group, True),
        ("runs", recovering, True),
        ("runs", new, False),
        ("runs", skipped, False),
    ]
    assert [result.status for result in error.value.results] == [
        "success",
        "success",
        "success",
        "failed",
    ]
    assert "already exists" in error.value.results[-1].error
    client, location, batches = published(c, "runs", skipped)
    assert batches == old_batches[skipped]
    assert (
        client.run("cat", f"{location}/batches/{batches[0]}/content/payload.txt")
        == "remote-old"
    )
    assert published(c, "runs", recovering)[2] != old_batches[recovering]
    assert read_json(group_dir / "record.json")["status"] == "completed"

    # Explicit manual collection overwrite still authorizes every selected run.
    attempts.clear()
    results = storage.upload(c, "groups", group, overwrite=True, root=tmp_path)
    assert all(result.status == "success" for result in results), (
        storage.format_results(results)
    )
    assert [allowed for _, _, allowed in attempts] == [True] * 4


def test_cli_transfer_failure_does_not_fail_computation(tmp_path, monkeypatch):
    c = cfg(tmp_path, storage={"remote": "archive:project"})
    monkeypatch.setattr(runner, "get", lambda name: simple_method())

    def fail(self, *args, **kwargs):
        raise RuntimeError("rclone unavailable")

    monkeypatch.setattr(storage.Rclone, "run", fail)
    atomic(tmp_path / "input.toml", dumps(c))
    cli = CliRunner()
    result = cli.invoke(
        app, ["train", "input.toml", "--id", "one", "--root", str(tmp_path)]
    )
    assert result.exit_code != 0 and isinstance(result.exception, storage.TransferError)
    assert "runs/one: failed" in result.output
    assert (
        read_json(tmp_path / "experiments/runs/one/record.json")["status"]
        == "completed"
    )
    result = cli.invoke(
        app,
        [
            "storage",
            "upload",
            "input.toml",
            "runs",
            "--id",
            "one",
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1 and "rclone unavailable" in result.output
    assert cli.invoke(app, ["storage", "download", "--help"]).exit_code == 0
