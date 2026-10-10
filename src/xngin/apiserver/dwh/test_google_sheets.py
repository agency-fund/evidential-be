import csv
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa

from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.google_sheets import PUBLIC_SHEET_TABLE, GoogleSheetsClient, SheetData
from xngin.apiserver.dwh.participant_metrics_queries import get_participant_metrics
from xngin.apiserver.dwh.queries import get_raw_metric_stats
from xngin.apiserver.exceptions_common import DwhConnectionError, LateValidationError
from xngin.apiserver.routers.admin import admin_api_types as aapi
from xngin.apiserver.routers.admin.admin_api_converters import api_dsn_to_settings_dwh, settings_dwh_to_api_dsn
from xngin.apiserver.routers.common_api_types import (
    Arm,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    Filter,
    FreqExperimentAnalysisResponse,
    PowerRequest,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import MetricPowerAnalysisMessageType, Relation
from xngin.apiserver.settings import GoogleSheetsDsn, RemoteDatabaseConfig

URL = "https://docs.google.com/spreadsheets/d/demo_sheet/edit#gid=42"
TAB = PUBLIC_SHEET_TABLE


@pytest.fixture
def sheets_http(mocker):
    mocker.patch("google.auth.default", side_effect=AssertionError("Public sheets need no credentials"))
    http = mocker.patch("xngin.apiserver.dwh.google_sheets.requests.Session").return_value
    values: list[list] = [
        ["participant_id", "region", "outcome", "converted"],
        ["001", "north", 10, True],
        [],
        ["002", "south", "", False],
        ["003", "north", 30, True],
    ]
    http.csv_tabs = {"42": values}

    def download(url, **kwargs):
        assert url == "https://docs.google.com/spreadsheets/d/demo_sheet/export"
        assert kwargs["params"]["format"] == "csv"
        tab_values = http.csv_tabs[kwargs["params"]["gid"]]
        assert "auth" not in kwargs
        assert "Authorization" not in kwargs["headers"]
        content = io.StringIO(newline="")
        csv.writer(content).writerows(tab_values)
        response = mocker.MagicMock(ok=True, headers={"Content-Type": "text/csv"})
        response.__enter__.return_value = response
        response.iter_content.return_value = [content.getvalue().encode()]
        return response

    http.get.side_effect = download
    return http, values


def test_dsn_roundtrip_and_storage():
    api_dsn = aapi.GoogleSheetsDsn(spreadsheet_url=URL)
    config = RemoteDatabaseConfig(type="remote", dwh=api_dsn_to_settings_dwh(api_dsn))
    reloaded = RemoteDatabaseConfig.model_validate_json(config.model_dump_json())
    assert settings_dwh_to_api_dsn(reloaded.dwh) == api_dsn


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.google.com/spreadsheets/d/demo",
        "https://evil.com/spreadsheets/d/demo",
        "https://docs.google.com.evil.com/spreadsheets/d/demo",
        "https://docs.google.com/spreadsheets/d/demo#gid=nope",
        "https://docs.google.com@localhost/spreadsheets/d/demo",
        "https://docs.google.com/spreadsheets/d/",
    ],
)
def test_only_spreadsheet_urls_are_accepted(url):
    with pytest.raises(ValueError):
        aapi.GoogleSheetsDsn(spreadsheet_url=url)
    with pytest.raises(ValueError):
        GoogleSheetsDsn(spreadsheet_url=url)


@pytest.mark.parametrize(
    "values",
    [[], [["id", "id"]], [["id", "ID"]], [["id", "has spaces"]], [["id", ""]], [["id"], ["1", "missing header"]]],
)
def test_bad_headers_are_rejected(values):
    with pytest.raises(LateValidationError):
        SheetData.from_values(values)


def test_queries_use_typed_cells_and_refresh_each_session(sheets_http):
    http, values = sheets_http
    metric = DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh:
        assert dwh.list_tables() == [TAB]
        table = dwh.inspect_table(TAB)
        assert isinstance(table.c.participant_id.type, sa.String)
        assert isinstance(table.c.converted.type, sa.Boolean)
        stats = dwh.run(get_raw_metric_stats, table, [metric], [])
        assert stats["rows__count"] == 3
        assert stats["outcome__count"] == 2
        assert stats["outcome__mean"] == 20
        assert stats["outcome__stddev"] == 10
        result = dwh.get_participants(
            TAB,
            select_columns={"participant_id", "outcome"},
            n=10,
            filters=[Filter(field_name="region", relation=Relation.INCLUDES, value=["north"])],
        )
        assert {row.participant_id for row in result.participants} == {"001", "003"}
        outcomes = dwh.run(get_participant_metrics, table, [metric], "participant_id", ["002"])
        assert outcomes[0].metric_values[0].metric_value is None
        assert http.get.call_count == 1  # All queries in a request use one consistent snapshot.
        assert http.trust_env is False
    # A manual edit becomes visible on the next analysis request, without a schema-cache refresh.
    values[3][2] = 50
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh:
        table = dwh.inspect_table(TAB)
        stats = dwh.run(get_raw_metric_stats, table, [metric], [])
        assert stats["outcome__count"] == 3
        assert stats["outcome__mean"] == 30
        assert stats["outcome__stddev"] == pytest.approx(np.std([10, 30, 50]))
    assert http.get.call_count == 2
    assert (
        http.get.call_args_list[0].kwargs["params"]["_evidential"] != http.get.call_args.kwargs["params"]["_evidential"]
    )


def test_empty_metric_is_numeric_and_zero_is_observed(sheets_http):
    _, values = sheets_http
    values[:] = [["id", "outcome"], ["001", ""], ["002", 0]]
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh:
        table = dwh.inspect_table(TAB)
        outcomes = dwh.run(
            get_participant_metrics,
            table,
            [DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)],
            "id",
            ["001", "002"],
        )
        assert {o.participant_id: o.metric_values[0].metric_value for o in outcomes} == {"001": None, "002": 0}
    values[2][1] = ""
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh:
        assert isinstance(dwh.inspect_table(TAB).c.outcome.type, sa.Double)


def test_export_matches_ids_after_reordering_and_preserves_other_columns(sheets_http):
    http, values = sheets_http
    values[1], values[4] = values[4], values[1]
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh:
        data = dwh.export_sheet_assignments(TAB, "participant_id", "demo-123", {"001": "Treatment", "003": "Control"})
    assert list(csv.reader(io.StringIO(data))) == [
        ["participant_id", "region", "outcome", "converted", "evidential_demo_123_arm"],
        ["003", "north", "30", "True", "Control"],
        ["", "", "", "", ""],
        ["002", "south", "", "False", ""],
        ["001", "north", "10", "True", "Treatment"],
    ]
    assert values[0] == ["participant_id", "region", "outcome", "converted"]
    http.post.assert_not_called()


def test_reexport_reuses_its_column(sheets_http):
    _, values = sheets_http
    values[0].append("evidential_demo_arm")
    values[1].append("Old treatment name")
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh:
        data = dwh.export_sheet_assignments(TAB, "participant_id", "demo", {"001": "Treatment"})
    exported = list(csv.reader(io.StringIO(data)))
    assert exported[0].count("evidential_demo_arm") == 1
    assert exported[1][-1] == "Treatment"
    assert values[1][-1] == "Old treatment name"


def test_experiment_csv_keeps_selected_columns_and_matches_original_ids(sheets_http):
    _, values = sheets_http
    values[:] = [
        ["notes", "outcome", "participant_id", "unused_metric", "evidential_other_arm"],
        ["internal", "", "001", 20, "Old control"],
        [],
        ["internal", 0, "002", 30, "Old treatment"],
        ["internal", 5, "003", 40, "Old control"],
    ]
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh:
        exported = dwh.export_sheet_assignments(
            TAB, "participant_id", "demo", {"001": "Treatment", "003": "Control"}, field_names={"outcome"}
        )
    assert list(csv.reader(io.StringIO(exported))) == [
        ["outcome", "participant_id", "evidential_demo_arm"],
        ["", "001", "Treatment"],
        ["", "", ""],
        ["0", "002", ""],
        ["5", "003", "Control"],
    ]
    assert values[0][0] == "notes"


def test_experiment_csv_rejects_missing_selected_columns(sheets_http):
    with (
        pytest.raises(LateValidationError, match=r"Experiment columns are missing.*missing_metric"),
        DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh,
    ):
        dwh.export_sheet_assignments(TAB, "participant_id", "demo", {}, field_names={"missing_metric"})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "unique"),
        ("missing", "missing"),
        ("blank_id", "participant ID"),
        ("missing_column", "column is missing"),
    ],
)
def test_export_rejects_invalid_participant_ids(sheets_http, mutation, match):
    http, values = sheets_http
    if mutation == "duplicate":
        values[4][0] = "001"
    elif mutation == "missing":
        values[1][0] = "new-id"
    elif mutation == "blank_id":
        values[4][0] = ""
    else:
        values[0][0] = "other_id"
    with pytest.raises(LateValidationError, match=match), DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh:
        dwh.export_sheet_assignments(TAB, "participant_id", "demo", {"001": "Treatment"})
    http.post.assert_not_called()


@pytest.mark.parametrize("status_code", [400, 403, 404, 429, 500])
def test_google_errors_are_actionable(sheets_http, mocker, status_code):
    http, _ = sheets_http
    http.get.side_effect = None
    response = mocker.MagicMock(ok=False, status_code=status_code)
    response.__enter__.return_value = response
    http.get.return_value = response
    with pytest.raises(DwhConnectionError, match="Google Sheets"):
        GoogleSheetsClient(URL)
    http.close.assert_called_once()


@pytest.mark.parametrize("gid", [42, None])
def test_invalid_tab_requires_reconnecting_instead_of_reading_another_tab(sheets_http, mocker, gid):
    http, _ = sheets_http
    response = mocker.MagicMock(ok=False, status_code=400)
    response.__enter__.return_value = response
    http.get.side_effect = None
    http.get.return_value = response
    url = URL if gid is not None else URL.split("#", maxsplit=1)[0]
    expected = "Connect Experiment tab" if gid is not None else "HTTP 400"
    with pytest.raises(DwhConnectionError, match=expected):
        GoogleSheetsClient(url)
    http.get.assert_called_once()
    http.close.assert_called_once()


def test_demo_limit_rejects_data_instead_of_truncating():
    with pytest.raises(LateValidationError, match="5,000"):
        SheetData.from_values([["id"], *[[str(i)] for i in range(5_001)]])
    with pytest.raises(LateValidationError, match="100"):
        SheetData.from_values([[f"col_{i}" for i in range(101)]])


def test_sign_in_page_is_an_access_error(sheets_http, mocker):
    http, _ = sheets_http
    response = mocker.MagicMock(ok=True, headers={"Content-Type": "text/html; charset=utf-8"})
    response.__enter__.return_value = response
    response.iter_content.return_value = [b"<html>Sign in</html>"]
    http.get.side_effect = None
    http.get.return_value = response
    with pytest.raises(DwhConnectionError, match="Anyone with the link: Viewer"):
        GoogleSheetsClient(URL)


@pytest.mark.parametrize("blank_outcomes", [False, True])
@pytest.mark.parametrize("cluster_key", [None, "region"])
def test_create_launch_export_import_and_analyze_a_sheet_experiment(sheets_http, aclient, blank_outcomes, cluster_key):
    http, values = sheets_http
    examples = Path(__file__).parents[4] / "tools"
    with (examples / "google_sheets_demo.csv").open(newline="") as source:
        observed_values = list(csv.reader(source))
    if blank_outcomes:
        values[:] = [observed_values[0].copy(), *[row[:2] + [""] * (len(row) - 2) for row in observed_values[1:]]]
    else:
        values[:] = observed_values
    headers = values[0].copy()
    metrics = [DesignSpecMetricRequest(field_name=name, metric_pct_change=0.1) for name in headers[2:]]
    if cluster_key is not None:
        metrics = [metrics[-1], *metrics[:-1]]  # Reproduce the initially blank onboarding primary metric.
    population_size = len(values) - 1
    org_id = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="Sheets demo")).data.id
    datasource_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=org_id, name="Demo sheet", dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL)
        ),
        connectivity_check=True,
    ).data.id
    assert aclient.inspect_datasource(datasource_id=datasource_id).data.tables == [TAB]
    fields = aclient.inspect_table_in_datasource(datasource_id=datasource_id, table_name=TAB).data
    assert "participant_id" in fields.detected_unique_id_fields
    assert all(
        field.data_type in {"bigint", "double precision"} for field in fields.fields if field.field_name in headers[2:]
    )
    power_request = PowerRequest(
        table_name=TAB,
        cluster_key=cluster_key,
        metrics=metrics,
        filters=[],
        n_arms=2,
        alpha=0.05,
        power=0.8,
    )
    initial_power = aclient.power_check(datasource_id=datasource_id, body=power_request).data
    for analysis in initial_power.analyses:
        column_index = headers.index(analysis.metric_spec.field_name)
        observed_count = sum(row[column_index] != "" for row in values[1:])
        assert analysis.metric_spec.available_n == population_size
        assert analysis.metric_spec.available_nonnull_n == observed_count
        if observed_count == 0:
            assert analysis.metric_spec.metric_baseline is None
            assert analysis.msg is not None
            assert analysis.msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT
            assert analysis.target_n is None
            if cluster_key is not None:
                assert analysis.metric_spec.icc is None
                assert analysis.metric_spec.avg_cluster_size == population_size / 4
                assert analysis.metric_spec.cv == 0
    # The wizard's maximum/custom-size choice works even without observations to estimate power.
    power = aclient.power_check(
        datasource_id=datasource_id,
        body=power_request.model_copy(
            update={"desired_n": population_size, "desired_n_clusters": 4 if cluster_key is not None else None}
        ),
    ).data
    if blank_outcomes or cluster_key is not None:
        assert power.analyses[0].pct_change_with_desired_n is None
        assert power.analyses[0].msg.type == MetricPowerAnalysisMessageType.INSUFFICIENT
    else:
        assert power.analyses[0].metric_spec.metric_baseline == pytest.approx(
            np.mean([float(row[2]) for row in values[1:]])
        )
    now = datetime.now(UTC)
    experiment_id = aclient.create_experiment(
        datasource_id=datasource_id,
        random_state=42,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type="freq_preassigned",
                experiment_name="Sheets demo",
                description="Demo",
                table_name=TAB,
                primary_key="participant_id",
                start_date=now - timedelta(days=1),
                end_date=now + timedelta(days=1),
                desired_n=population_size if cluster_key is None else None,
                desired_n_clusters=4 if cluster_key is not None else None,
                cluster_key=cluster_key,
                metrics=metrics,
                strata=[],
                filters=[],
                arms=[
                    Arm(arm_name="Control", arm_description="Control"),
                    Arm(arm_name="Treatment", arm_description="Treatment"),
                ],
            ),
            power_analyses=power,
        ),
    ).data.experiment_id
    reads_before_launch = http.get.call_count
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)
    assert http.get.call_count == reads_before_launch  # Launch never contacts Google or writes back.
    assert values[0] == headers
    values[1][0] = "changed-id"
    invalid_export = aclient.get_experiment_assignments_as_csv_for_ui(
        datasource_id=datasource_id, experiment_id=experiment_id, raise_if_not_default_status=False
    )
    assert invalid_export.status == 422
    assert "missing" in str(invalid_export.data)
    values[1][0] = "p001"
    download = aclient.get_experiment_assignments_as_csv_for_ui(
        datasource_id=datasource_id, experiment_id=experiment_id
    )
    # Import into a separate tab, preserving the source tab and reconnecting by its new gid.
    imported = list(csv.reader(io.StringIO(b"".join(download.data).decode())))
    expected_fields = {"participant_id", *(metric.field_name for metric in metrics)}
    if cluster_key is not None:
        expected_fields.add(cluster_key)
    indices = [index for index, name in enumerate(headers) if name in expected_fields]
    assert [row[:-1] for row in imported] == [[row[index] for index in indices] for row in values]
    assert values[0] == headers
    source_tab = [row.copy() for row in values]
    http.csv_tabs["99"] = imported
    aclient.update_datasource(
        datasource_id=datasource_id,
        body=aapi.UpdateDatasourceRequest(dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL.replace("gid=42", "gid=99"))),
    )
    values = imported
    header = f"evidential_{experiment_id.replace('-', '_')}_arm"
    assert values[0][-1] == header
    assert {row[-1] for row in values[1:]} == {"Control", "Treatment"}
    if cluster_key is not None:
        assert all(
            len({row[-1] for row in values[1:] if row[1] == region}) == 1 for region in {row[1] for row in values[1:]}
        )
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)
    waiting = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert isinstance(waiting, FreqExperimentAnalysisResponse)
    assert waiting.num_participants == population_size
    onboarding_waiting = next(m for m in waiting.metric_analyses if m.metric_name == "onboarded_within_1_week")
    assert all(arm.num_missing_values == -1 for arm in onboarding_waiting.arm_analyses)
    if blank_outcomes:
        assert all(arm.num_missing_values == -1 for metric in waiting.metric_analyses for arm in metric.arm_analyses)
    # Fill the initially blank onboarding column with observed 1/0 values using the existing numeric path.
    observed_by_id = {
        row[0]: dict(zip(headers[2:], [*row[2:-1], str(int(row[0][1:]) % 2)], strict=True))
        for row in observed_values[1:]
    }
    for row in values[1:]:
        for metric in metrics:
            row[values[0].index(metric.field_name)] = observed_by_id[row[0]][metric.field_name]
    minutes_index = values[0].index("minutes_on_site_last_7_days")
    if blank_outcomes:
        values[1][minutes_index] = "0"  # Zero is observed, even when the other outcomes started blank.

    def save_analysis():
        snapshot_id = aclient.create_snapshot(
            organization_id=org_id, datasource_id=datasource_id, experiment_id=experiment_id
        ).data.id
        snapshot = aclient.get_snapshot(
            organization_id=org_id,
            datasource_id=datasource_id,
            experiment_id=experiment_id,
            snapshot_id=snapshot_id,
        ).data.snapshot
        assert snapshot.status == "success", snapshot.details
        return snapshot_id, FreqExperimentAnalysisResponse.model_validate(snapshot.data)

    before_id, before = save_analysis()
    # Reorder the spreadsheet and update treatment outcomes as a presenter would do in Sheets.
    values[1:] = reversed(values[1:])
    for row in values[1:]:
        if row[-1] == "Treatment":
            row[minutes_index] = str(float(row[minutes_index]) + 100)
    after_id, after = save_analysis()
    assert isinstance(before, FreqExperimentAnalysisResponse)
    assert isinstance(after, FreqExperimentAnalysisResponse)
    assert [metric.metric_name for metric in before.metric_analyses] == [metric.field_name for metric in metrics]
    assert all(arm.num_missing_values == 0 for metric in before.metric_analyses for arm in metric.arm_analyses)
    onboarding = next(m for m in before.metric_analyses if m.metric_name == "onboarded_within_1_week")
    observed_rates = {
        arm: np.mean([float(row[-2]) for row in values[1:] if row[-1] == arm]) for arm in ("Control", "Treatment")
    }
    for arm in onboarding.arm_analyses:
        expected = (
            observed_rates["Control"] if arm.is_baseline else observed_rates["Treatment"] - observed_rates["Control"]
        )
        assert arm.estimate == pytest.approx(expected)
    before_minutes = next(m for m in before.metric_analyses if m.metric_name == "minutes_on_site_last_7_days")
    after_minutes = next(m for m in after.metric_analyses if m.metric_name == "minutes_on_site_last_7_days")
    before_treatment = next(a for a in before_minutes.arm_analyses if a.arm_name == "Treatment")
    after_treatment = next(a for a in after_minutes.arm_analyses if a.arm_name == "Treatment")
    assert after_treatment.estimate == pytest.approx(before_treatment.estimate + 100)
    for previous, current in zip(before.metric_analyses, after.metric_analyses, strict=True):
        if previous.metric_name == "minutes_on_site_last_7_days":
            continue
        assert {arm.arm_name: arm.estimate for arm in current.arm_analyses} == pytest.approx({
            arm.arm_name: arm.estimate for arm in previous.arm_analyses
        })
    saved = aclient.list_snapshots(
        organization_id=org_id,
        datasource_id=datasource_id,
        experiment_id=experiment_id,
        status_=[aapi.SnapshotStatus.SUCCESS],
    ).data.items
    assert [snapshot.id for snapshot in saved] == [after_id, before_id]
    assert saved[0].updated_at > saved[1].updated_at
    assert FreqExperimentAnalysisResponse.model_validate(saved[0].data) == after
    assert FreqExperimentAnalysisResponse.model_validate(saved[1].data) == before
    http.post.assert_not_called()
    assert http.csv_tabs["42"] == source_tab
    assert http.get.call_args.kwargs["params"]["gid"] == "99"
