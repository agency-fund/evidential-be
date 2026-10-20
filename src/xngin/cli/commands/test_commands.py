from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from xngin.cli.commands.create_testing_dwh import _Template  # noqa: PLC2701
from xngin.cli.main import app

_WAREHOUSES = Path(__file__).resolve().parents[2] / "apiserver/testdata/warehouses.json"


def test_warehouses_template_lists_existing_files():
    template = _Template.model_validate_json(_WAREHOUSES.read_text())

    assert [t.table_name for t in template.tables] == ["dwh", "wide_dwh", "clustered_dwh"]
    for table in template.tables:
        assert (_WAREHOUSES.parent / table.src).is_file()


def test_template_rejects_unknown_keys():
    with pytest.raises(ValidationError, match="view"):
        _Template.model_validate_json('{"tables": [{"src": "a.csv", "table_name": "a", "view": ["b"]}]}')


def test_template_conflicts_with_table_name_option():
    result = CliRunner().invoke(
        app, ["create-testing-dwh", "--dsn", "sqlite://", "--template", str(_WAREHOUSES), "--table-name", "x"]
    )

    assert result.exit_code == 2
    assert "cannot be combined" in result.output


def test_template_ignores_table_name_from_environment():
    result = CliRunner().invoke(
        app,
        ["create-testing-dwh", "--dsn", "sqlite://", "--template", str(_WAREHOUSES)],
        env={"XNGIN_CLI_TABLE_NAME": "x"},
    )

    # Fails only later, on the unsupported database.
    assert result.exit_code == 2
    assert "Unrecognized database driver" in result.output


def test_drop_database_requires_names_or_prefix():
    result = CliRunner().invoke(app, ["drop-database", "--dsn", "postgresql://localhost/postgres"])

    assert result.exit_code == 2
    assert "--prefix" in result.output
