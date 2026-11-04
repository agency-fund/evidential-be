"""pytest plugin choosing how many pytest-xdist workers -n auto starts.

Every worker imports the application and collects every test, which costs a few seconds of CPU each, so more workers
only pay off for runs of many tests. It is registered with -p in pyproject.toml because pytest-xdist asks for the
number of workers before conftests are loaded.
"""

import os

import pytest
from xdist.plugin import pytest_xdist_auto_num_workers as xdist_auto_num_workers

# Measured on a 16-core laptop: the suite took as long with 8 workers as with 16, using half the CPU.
MAX_WORKERS = 8


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Runs named tests serially and otherwise reserves a core for the rest of the system, up to MAX_WORKERS.

    The tests that -k and --lf select aren't known until they are collected, so those runs get 2 workers, which is
    almost as fast as running serially for a few tests and still splits a run of many.
    """
    if os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS"):
        return None  # Let pytest-xdist honor the override.
    if any("::" in arg for arg in config.args):
        return 0
    if config.option.keyword or config.option.lf:
        return 2
    return max(1, min(int(xdist_auto_num_workers(config)) - 1, MAX_WORKERS))
