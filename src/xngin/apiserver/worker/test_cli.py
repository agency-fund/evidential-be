import contextlib
import inspect
import json
from datetime import timedelta
from pathlib import Path

from xngin.apiserver.worker import cli

RAILWAY_CONFIG = Path(__file__).parents[4] / "railway.worker.json"


def test_railway_draining_covers_default_warehouse_timeouts():
    defaults = inspect.signature(cli.run).parameters
    draining = json.loads(RAILWAY_CONFIG.read_text())["deploy"]["drainingSeconds"]
    assert draining >= defaults["dwh_pull_timeout"].default
    assert draining >= defaults["snapshot_dwh_timeout"].default


def test_broken_warehouse_pulls_back_off_to_an_hour_within_the_pull_interval():
    # A pull that keeps failing must not stay on the eager cadence, but must still retry well within the pull
    # interval so a repaired warehouse is picked up promptly.
    delays = [cli.DWH_PULL_BACKOFF.delay_after(failures) for failures in range(1, 10)]
    assert delays == sorted(delays)
    assert delays[0] == timedelta(minutes=1)
    assert delays[-1] == timedelta(hours=1)
    assert delays[-1] < timedelta(seconds=inspect.signature(cli.run).parameters["dwh_pull_interval"].default)


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
