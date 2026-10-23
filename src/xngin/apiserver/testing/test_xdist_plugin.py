from types import SimpleNamespace

import pytest

from xngin.apiserver.testing import xdist_plugin


@pytest.mark.parametrize(
    ("cores", "args", "keyword", "lf", "override", "expected"),
    [
        (16, ["src"], "", False, None, 8),
        (6, ["src"], "", False, None, 5),
        (1, ["src"], "", False, None, 1),
        (16, ["src/xngin/test_a.py::test_one"], "", False, None, 0),
        (16, ["src"], "crypt", False, None, 2),
        (16, ["src"], "", True, None, 2),
        (16, ["src"], "", False, "3", None),
    ],
    ids=["capped", "reserves_a_core", "one_core", "named_test", "keyword", "last_failed", "xdist_override"],
)
def test_auto_num_workers(monkeypatch, cores, args, keyword, lf, override, expected):
    monkeypatch.setattr(xdist_plugin, "xdist_auto_num_workers", lambda config: cores)
    if override is None:
        monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    else:
        monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", override)
    config = SimpleNamespace(args=args, option=SimpleNamespace(keyword=keyword, lf=lf))

    assert xdist_plugin.pytest_xdist_auto_num_workers(config) == expected  # type: ignore[arg-type]
