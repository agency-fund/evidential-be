"""Tests for `xngin-cli seed-demo`. The seed itself is covered in apiserver/storage/test_demo_seed.py."""

import os

import pytest
from typer.testing import CliRunner

from xngin.apiserver.storage import demo_seed
from xngin.cli.main import app

runner = CliRunner()


def _output(result) -> str:
    """Returns the command output with whitespace collapsed, since Rich wraps long lines to the terminal width."""
    return " ".join(result.output.split())


@pytest.fixture(name="database_url")
def fixture_database_url() -> str:
    # The seed is mocked in these tests, so the command only needs a URL it can build an engine from.
    return os.environ["DATABASE_URL"]


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"XNGIN_DEMO_DWH_DSN": "postgresql+psycopg://localhost/dwh"},
        {"XNGIN_DEMO_ORG_EMAILS": "a@example.com"},
        {"XNGIN_DEMO_DWH_DSN": "postgresql+psycopg://localhost/dwh", "XNGIN_DEMO_ORG_EMAILS": " , "},
    ],
)
def test_seed_demo_does_nothing_unless_configured(database_url: str, mocker, env: dict[str, str]):
    seed = mocker.patch.object(demo_seed, "seed_for_emails")

    result = runner.invoke(app, ["seed-demo", "--database-url", database_url], env=env)

    assert result.exit_code == 0
    assert "nothing to do" in _output(result)
    seed.assert_not_called()


def test_seed_demo_reports_each_email(database_url: str, mocker):
    seed = mocker.patch.object(
        demo_seed, "seed_for_emails", return_value={"a@example.com": "created", "b@example.com": "no such user"}
    )

    result = runner.invoke(
        app,
        ["seed-demo", "--database-url", database_url],
        env={
            "XNGIN_DEMO_DWH_DSN": "postgresql+psycopg://localhost/dwh",
            "XNGIN_DEMO_ORG_EMAILS": "a@example.com, b@example.com",
        },
    )

    assert result.exit_code == 0
    # Blank entries and surrounding spaces in the allowlist are dropped.
    assert seed.call_args.args[1:] == ("postgresql+psycopg://localhost/dwh", ["a@example.com", "b@example.com"])
    assert "seed-demo: a@example.com: created" in _output(result)
    assert "seed-demo: b@example.com: no such user" in _output(result)


def test_seed_demo_exits_zero_when_seeding_raises(database_url: str, mocker):
    """An unexpected error is reported but never fails the command, so it cannot block a Railway deploy."""
    mocker.patch.object(demo_seed, "seed_for_emails", side_effect=RuntimeError("boom"))

    result = runner.invoke(
        app,
        ["seed-demo", "--database-url", database_url],
        env={"XNGIN_DEMO_DWH_DSN": "postgresql+psycopg://localhost/dwh", "XNGIN_DEMO_ORG_EMAILS": "a@example.com"},
    )

    assert result.exit_code == 0
    assert "seed-demo failed" in _output(result)
