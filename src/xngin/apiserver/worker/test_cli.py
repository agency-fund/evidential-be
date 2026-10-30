import contextlib
import inspect
import json
from pathlib import Path

from xngin.apiserver.worker import cli

RAILWAY_CONFIG = Path(__file__).parents[4] / "railway.worker.json"


def test_railway_draining_covers_default_warehouse_timeouts():
    defaults = inspect.signature(cli.run).parameters
    draining = json.loads(RAILWAY_CONFIG.read_text())["deploy"]["drainingSeconds"]
    assert draining >= defaults["dwh_pull_timeout"].default
    assert draining >= defaults["snapshot_dwh_timeout"].default


def test_escaping_exception_still_exits_immediately(monkeypatch):
    exits: list[int] = []

    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    for setup in (cli.customlogging, cli.sentry, cli.secretservice):
        monkeypatch.setattr(setup, "setup", lambda: None)
    monkeypatch.setattr(cli.database, "setup", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(cli, "run_worker", fail)
    monkeypatch.setattr(cli.os, "_exit", exits.append)
    cli.run()
    assert exits == [1]
