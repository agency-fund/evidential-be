"""Tests for queries.py."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DataError

from xngin.apiserver.conftest import DbType, get_queries_test_uri
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.queries import get_stats_on_metrics
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import DesignSpecMetric, DesignSpecMetricRequest
from xngin.apiserver.routers.common_enums import MetricType
from xngin.apiserver.settings import Dsn

pytest_plugins = ("xngin.apiserver.dwh.dwh_test_support",)


def test_get_stats_on_missing_metric_raises_error(queries_dwh_session, shared_sample_tables):
    with pytest.raises(LateValidationError) as exc:
        get_stats_on_metrics(
            queries_dwh_session,
            shared_sample_tables.sample_table,
            [DesignSpecMetricRequest(field_name="missing_col", metric_pct_change=0.1)],
            filters=[],
        )
    assert "Missing metrics (check your Datasource configuration): {'missing_col'}" in str(exc.value)


def test_get_stats_on_integer_metric(queries_dwh_session, shared_sample_tables):
    """Test would fail on postgres and redshift without a cast to float for different reasons."""
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        filters=[],
    )

    expected = DesignSpecMetric(
        field_name="int_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=41.666666666666664,
        metric_stddev=47.76563153100307,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    # PG: assertion would fail due to a float vs decimal.Decimal comparison.
    # RS: assertion would fail due to avg() on int types keeps them as integers.
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_nullable_integer_metric(queries_dwh_session, shared_sample_tables):
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_nullable_table,
        [DesignSpecMetricRequest(field_name="int_col", metric_pct_change=0.1)],
        filters=[],
    )

    expected = DesignSpecMetric(
        field_name="int_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=2.0,
        metric_stddev=1.0,
        available_nonnull_n=2,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_boolean_metric(queries_dwh_session, shared_sample_tables):
    """Test would fail on postgres and redshift without casting to int to float."""
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="bool_col", metric_pct_change=0.1)],
        filters=[],
    )

    expected = DesignSpecMetric(
        field_name="bool_col",
        metric_type=MetricType.BINARY,
        metric_baseline=0.6666666666666666,
        metric_stddev=None,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def test_get_stats_on_numeric_metric(queries_dwh_session, shared_sample_tables):
    rows = get_stats_on_metrics(
        queries_dwh_session,
        shared_sample_tables.sample_table,
        [DesignSpecMetricRequest(field_name="float_col", metric_pct_change=0.1)],
        filters=[],
    )

    expected = DesignSpecMetric(
        field_name="float_col",
        metric_type=MetricType.NUMERIC,
        metric_baseline=2.492,
        metric_stddev=0.6415751449882287,
        available_nonnull_n=3,
        available_n=3,
    )
    assert len(rows) == 1
    actual = rows[0]
    numeric_fields = {
        "metric_baseline",
        "metric_stddev",
        "available_nonnull_n",
        "available_n",
    }
    assert actual.field_name == expected.field_name
    assert actual.metric_type == expected.metric_type
    # pytest.approx does a reasonable fuzzy comparison of floats for non-nested dictionaries.
    assert actual.model_dump(include=numeric_fields) == pytest.approx(expected.model_dump(include=numeric_fields))


def _pg_dsn(search_path: str) -> Dsn:
    """Builds a Dsn from the test URI with the given search_path. Skips the test if not PG.

    Connects to the 'postgres' system database rather than the test URI db so we don't interfere
    with the module-scoped queries_dwh_engine fixture.
    """
    test_db = get_queries_test_uri()
    if test_db.db_type != DbType.PG:
        pytest.skip("search_path is only supported for PostgreSQL")

    url = test_db.connect_url
    return Dsn(
        driver=url.drivername,
        host=url.host,
        port=url.port,
        user=url.username,
        password=url.password,
        dbname="postgres",
        sslmode="disable",
        search_path=search_path,
    )


@pytest.mark.parametrize(
    "search_path",
    [
        "public",
        'foo"bar',
        "public, myschema",
    ],
)
async def test_search_path_is_set_on_session(search_path):
    """Verify that DwhSession sets search_path for every new PostgreSQL connection."""
    async with DwhSession(_pg_dsn(search_path)) as dwh:
        result = await asyncio.to_thread(dwh.session.execute, text("SELECT current_setting('search_path')"))
    assert result.scalar() == search_path


async def test_search_path_injection_attempt_is_rejected():
    """Check that a malicious search_path string is safely passed as a value, not interpreted as SQL."""
    # Postgres rejects the value as invalid search_path list syntax, rather than interpolating.
    with pytest.raises(DataError, match="invalid value for parameter"):
        async with DwhSession(_pg_dsn('public"; DROP TABLE users; --')) as dwh:
            await asyncio.to_thread(dwh.session.execute, text("SELECT 1"))
