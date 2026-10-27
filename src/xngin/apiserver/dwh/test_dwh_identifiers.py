"""Tests of the table, schema and column names DwhSession can use, on Postgres, Redshift and BigQuery."""

from dataclasses import astuple

import pytest

from xngin.apiserver.dwh import dwh_utils
from xngin.apiserver.routers.common_api_types import (
    DesignSpecMetricRequest,
    Filter,
)
from xngin.apiserver.routers.common_enums import Relation
from xngin.apiserver.testing.dwh_identifiers_helpers import (
    BQ_QUOTED_COLUMNS,
    PG_QUOTED_COLUMNS,
)


@pytest.mark.parametrize("name", sorted({*astuple(PG_QUOTED_COLUMNS), *astuple(BQ_QUOTED_COLUMNS)}))
def test_column_names_that_need_quoting_are_accepted_as_fields(name: str):
    assert Filter(field_name=name, relation=Relation.INCLUDES, value=["a"]).field_name == name
    assert DesignSpecMetricRequest(field_name=name, metric_pct_change=0.1).field_name == name


@pytest.mark.parametrize(
    "name",
    ["plain", "has space", "dash-name", "tëst_tablé", "名前", "Ω mixed-Ünï 1", "x" * 1024, "é" * 512, "t -- c"],
)
def test_bigquery_table_names_that_bigquery_allows_are_accepted(name: str):
    """Names that look like SQL, such as "t -- c", are still just names inside the backtick-quoted identifier."""
    assert dwh_utils.is_bigquery_table_name(name)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("", id="empty"),
        pytest.param("x" * 1025, id="over-1024-bytes"),
        pytest.param("é" * 513, id="over-1024-bytes-multibyte"),
        pytest.param("t` UNION ALL SELECT 1 FROM secret `", id="backtick"),
        pytest.param("t\\`x", id="backslash"),
        pytest.param("t\nx", id="newline"),
        pytest.param("t\x00x", id="nul"),
        pytest.param("t*", id="wildcard"),
        pytest.param("other_dataset.t", id="dot"),
        pytest.param("t; SELECT 1", id="semicolon"),
        pytest.param("t' OR '1'='1", id="single-quote"),
        pytest.param("\u202et", id="bidi-override"),
    ],
)
def test_bigquery_table_names_bigquery_cannot_create_are_rejected(name: str):
    assert not dwh_utils.is_bigquery_table_name(name)
