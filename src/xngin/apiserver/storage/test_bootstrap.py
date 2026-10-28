"""Tests of the samples bootstrap.py creates for a dev environment's first user that need no warehouse.

The stress test data sources must also work end to end: creating their experiments queries the warehouse, and a failure
would break the first sign-in. test_bootstrap_postgres.py and test_bootstrap_bigquery.py check that.
"""

from xngin.apiserver.dwh import dwh_utils
from xngin.apiserver.testing import stress_dwh_def


def test_the_stress_rows_are_reproducible():
    rows = stress_dwh_def.generate_rows()
    assert rows == stress_dwh_def.generate_rows()
    assert [row["id"] for row in rows] == list(range(1, stress_dwh_def.ROW_COUNT + 1))


def test_the_stress_bigquery_names_are_ones_bigquery_allows():
    assert dwh_utils.is_bigquery_table_name(stress_dwh_def.BIGQUERY.table)
    for column in stress_dwh_def.BIGQUERY.columns:
        assert "." not in column.name
        assert "$" not in column.name
    row = stress_dwh_def.bigquery_row(stress_dwh_def.generate_rows()[0])
    assert set(row) == {column.name for column in stress_dwh_def.BIGQUERY.columns}


def test_the_redshift_experiment_uses_the_lowercase_names_redshift_lists():
    """With enable_case_sensitive_identifier off, Redshift stores every name in lowercase, even quoted ones."""
    redshift_names = {column.name.lower() for column in stress_dwh_def.REDSHIFT.columns}
    experiment = stress_dwh_def.REDSHIFT_EXPERIMENT
    assert {experiment.primary_key, experiment.metric, experiment.filter_field, experiment.strata} <= redshift_names
    schema = stress_dwh_def.REDSHIFT.schema
    assert schema is not None
    assert f'"{schema.lower()}"' == stress_dwh_def.REDSHIFT_SEARCH_PATH
