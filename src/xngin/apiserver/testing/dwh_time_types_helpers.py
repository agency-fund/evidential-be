"""Checks, through the admin API, of the date and time columns of a stress test table loaded into a warehouse.

See xngin/apiserver/dwh/test_dwh_time_types_*.py.
"""

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

from xngin.apiserver.routers.admin.admin_api_types import CreateDatasourceRequest, CreateOrganizationRequest
from xngin.apiserver.routers.common_api_types import (
    Arm,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    Filter,
    PowerRequest,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import DataType, Relation
from xngin.apiserver.settings import BqDsn, Dsn
from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.helpers import convert_dwh_to_create_api_dsn
from xngin.apiserver.testing.stress_dwh_def import StressTable

type Row = dict[str, object]


@dataclass(frozen=True)
class TimeFilter:
    """A filter on the column with the logical key, and which of the generated rows it should match."""

    key: str
    relation: Relation
    value: list[str | None]
    matches: Callable[[Row], bool]


def _between(value: object, low: object, high: object) -> bool:
    return value is not None and (low is None or value >= low) and (high is None or value <= high)  # type: ignore[operator]


_UTC_MARCH = dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
_UTC_JUNE = dt.datetime(2026, 6, 1, tzinfo=dt.UTC)

# Filter values are UTC; the time zone-aware column holds instants that the source files write in several offsets.
TIME_FILTERS = [
    TimeFilter(
        "birth_date",
        Relation.BETWEEN,
        [None, "1969-12-31"],
        lambda r: _between(r["birth_date"], None, dt.date(1969, 12, 31)),
    ),
    TimeFilter("birth_date", Relation.INCLUDES, ["2000-02-29"], lambda r: r["birth_date"] == stress_dwh_def.LEAP_DAY),
    TimeFilter(
        "birth_date",
        Relation.BETWEEN,
        ["1950-01-01", "1979-12-31", None],
        lambda r: r["birth_date"] is None or _between(r["birth_date"], dt.date(1950, 1, 1), dt.date(1979, 12, 31)),
    ),
    TimeFilter(
        "appointment",
        Relation.BETWEEN,
        [None, "1969-12-31T23:59:59"],
        lambda r: _between(r["appointment"], None, dt.datetime(1969, 12, 31, 23, 59, 59)),
    ),
    TimeFilter(
        "appointment",
        Relation.BETWEEN,
        ["2038-01-19T03:14:08", None],
        lambda r: _between(r["appointment"], dt.datetime(2038, 1, 19, 3, 14, 8), None),
    ),
    TimeFilter(
        "appointment", Relation.INCLUDES, ["2026-03-08T02:30:00"], lambda r: r["appointment"] == stress_dwh_def.DST_GAP
    ),
    TimeFilter(
        "last_login",
        Relation.BETWEEN,
        ["2026-03-01T00:00:00Z", "2026-06-01T00:00:00+00:00"],
        lambda r: _between(r["last_login"], _UTC_MARCH, _UTC_JUNE),
    ),
    # A value without a time zone is taken to be UTC, whatever the warehouse session's time zone is. Only KNOWN_LOGIN
    # falls within these two hours, so taking them in another time zone would miss it.
    TimeFilter(
        "last_login",
        Relation.BETWEEN,
        ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"],
        lambda r: r["last_login"] == stress_dwh_def.KNOWN_LOGIN,
    ),
    TimeFilter(
        "last_login",
        Relation.BETWEEN,
        ["2026-07-01T00:00:00", "2026-07-01T01:00:00"],
        lambda r: r["last_login"] == stress_dwh_def.KNOWN_LOGIN,
    ),
    TimeFilter(
        "last_login",
        Relation.BETWEEN,
        ["2026-03-01T00:00:00", "2026-06-01T00:00:00"],
        lambda r: _between(r["last_login"], _UTC_MARCH, _UTC_JUNE),
    ),
]


def check_time_fields(
    aclient: AdminAPIClient, dwh_config: Dsn | BqDsn, table: StressTable, *, name: Callable[[str], str] = str
) -> None:
    """Checks a data source holding table: the types inspection reports, filters on them, and an experiment.

    name maps a column's name in table to the name the warehouse lists it under, e.g. in lowercase for Redshift.
    """

    def field(key: str) -> str:
        return name(table.column(key).name)

    rows = stress_dwh_def.generate_rows()
    org_id = aclient.create_organizations(body=CreateOrganizationRequest(name="time types")).data.id
    datasource_id = aclient.create_datasource(
        body=CreateDatasourceRequest(
            organization_id=org_id, name="time types", dsn=convert_dwh_to_create_api_dsn(dwh_config)
        )
    ).data.id
    table_name = name(table.table)

    inspected = aclient.inspect_table_in_datasource(datasource_id=datasource_id, table_name=table_name).data
    data_types = {f.field_name: f.data_type for f in inspected.fields}
    assert {
        key: data_types.get(field(key))
        for key in ("signup_date", "birth_date", "last_login", "appointment", "check_in", "check_in_tz")
        if any(c.key == key for c in table.columns)
    } == {
        "signup_date": DataType.DATE,
        "birth_date": DataType.DATE,
        "last_login": DataType.TIMESTAMP_WITH_TIMEZONE,
        "appointment": DataType.TIMESTAMP_WITHOUT_TIMEZONE,
        # Times of day have no DataType, so they are not offered as fields.
        "check_in": None,
        **({"check_in_tz": None} if any(c.key == "check_in_tz" for c in table.columns) else {}),
    }

    metric = DesignSpecMetricRequest(field_name=field("visits"), metric_pct_change=0.5)
    for time_filter in TIME_FILTERS:
        filter_ = Filter(field_name=field(time_filter.key), relation=time_filter.relation, value=time_filter.value)
        power = aclient.power_check(
            datasource_id=datasource_id,
            body=PowerRequest(table_name=table_name, metrics=[metric], filters=[filter_], n_arms=2),
        ).data
        expected = sum(time_filter.matches(row) for row in rows)
        assert power.analyses[0].metric_spec.available_n == expected, (time_filter.key, time_filter.value)

    # An experiment assigns every participant who logged in since a UTC instant.
    since = dt.datetime.fromisoformat(stress_dwh_def.LOGGED_IN_SINCE)
    expected = sum(_between(row["last_login"], since, None) for row in rows)
    now = dt.datetime.now(dt.UTC)
    created = aclient.create_experiment(
        datasource_id=datasource_id,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_name="time types",
                description="Filters on a timestamp",
                table_name=table_name,
                primary_key=field("id"),
                start_date=now,
                end_date=now + dt.timedelta(days=7),
                arms=[
                    Arm(arm_name="control", arm_description="Control"),
                    Arm(arm_name="treatment", arm_description="Treatment"),
                ],
                filters=[
                    Filter(
                        field_name=field("last_login"),
                        relation=Relation.BETWEEN,
                        value=[stress_dwh_def.LOGGED_IN_SINCE, None],
                    )
                ],
                strata=[],
                metrics=[metric],
                desired_n=expected,
            )
        ),
        random_state=42,
    ).data
    assert created.assign_summary is not None
    assert created.assign_summary.sample_size == expected
