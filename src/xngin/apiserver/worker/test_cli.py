import inspect
import json
from pathlib import Path

from xngin.apiserver.worker import cli

RAILWAY_CONFIG = Path(__file__).parents[4] / "railway.worker.json"


def test_railway_draining_covers_default_warehouse_timeouts():
    defaults = inspect.signature(cli.run).parameters
    draining = json.loads(RAILWAY_CONFIG.read_text())["deploy"]["drainingSeconds"]
    assert draining >= defaults["snapshot_dwh_timeout"].default
