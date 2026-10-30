import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from urllib.parse import quote

import gspread
import numpy as np
import pytest
import requests
import sqlalchemy as sa
from google.auth.credentials import AnonymousCredentials
from google.auth.exceptions import RefreshError
from sqlalchemy.orm import Session

from xngin.apiserver import flags
from xngin.apiserver.dwh.dwh_session import DwhSession
from xngin.apiserver.dwh.google_sheets import MAX_SHEET_BYTES, GoogleSheetsClient, SheetData, SheetsHTTPClient
from xngin.apiserver.dwh.participant_metrics_queries import get_participant_metrics
from xngin.apiserver.dwh.queries import get_raw_metric_stats
from xngin.apiserver.exceptions_common import DwhConnectionError, LateValidationError
from xngin.apiserver.limits import MAX_LENGTH_OF_NAME_VALUE
from xngin.apiserver.routers.admin import admin_api_types as aapi
from xngin.apiserver.routers.admin.admin_api_converters import settings_dwh_to_api_dsn
from xngin.apiserver.routers.common_api_types import (
    Arm,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    Filter,
    FreqExperimentAnalysisResponse,
    PowerRequest,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import ExperimentState, MetricPowerAnalysisMessageType, Relation
from xngin.apiserver.settings import GoogleSheetsDsn, RemoteDatabaseConfig
from xngin.apiserver.sqla import tables

URL = "https://docs.google.com/spreadsheets/d/demo_sheet/edit#gid=42"
TAB = "Evidential Demo Sheet - Data"


@pytest.fixture
def sheets_api(mocker):
    mocker.patch.object(flags, "GOOGLE_SHEETS_CREDENTIALS_FILE", "/configured/service-account.json")
    values = [
        ["participant_id", "region", "outcome", "converted"],
        ["001", "north", "10", "TRUE"],
        [],
        ["002", "south", "", "FALSE"],
        ["003", "north", "30", "TRUE"],
    ]
    api = SimpleNamespace(
        tabs={42: values},
        titles={42: "Data"},
        metadata={},
        grids={},
        fail_request_at=None,
        selected_gid=None,
        client=mocker.MagicMock(),
        writer=mocker.MagicMock(),
        spreadsheet=mocker.MagicMock(title="Evidential Demo Sheet"),
    )
    api.client.http_client.auth.service_account_email = "reader@evidential-tests.iam.gserviceaccount.com"
    api.writer.http_client.auth = api.client.http_client.auth.with_scopes.return_value
    api.writer.open_by_key.return_value = api.spreadsheet
    api.authorize = mocker.patch("xngin.apiserver.dwh.google_sheets.gspread.authorize", return_value=api.writer)

    def metadata(**_):
        return {
            "sheets": [
                {
                    "properties": {"sheetId": gid, "title": api.titles.get(gid, f"Outcomes {gid}")},
                    "developerMetadata": deepcopy(api.metadata.get(gid, [])),
                }
                for gid in api.tabs
            ]
        }

    def batch_update(body):
        # Publish only once all requests validate, just like spreadsheets.batchUpdate.
        tabs, titles = deepcopy(api.tabs), deepcopy(api.titles)
        metadata, grids = deepcopy(api.metadata), deepcopy(api.grids)
        for index, request in enumerate(body["requests"]):
            if api.fail_request_at == index:
                raise google_api_error(400)
            if "addSheet" in request:
                properties = request["addSheet"]["properties"]
                gid, title = properties["sheetId"], properties["title"]
                if gid in tabs or title in titles.values():
                    raise google_api_error(400)
                assert 0 <= gid < 2**31
                grid = properties["gridProperties"]
                assert grid["rowCount"] > 0 and grid["columnCount"] > 0
                tabs[gid] = [[""] * grid["columnCount"] for _ in range(grid["rowCount"])]
                titles[gid], grids[gid] = title, grid
            elif "updateCells" in request:
                update = request["updateCells"]
                start = update["start"]
                gid = start["sheetId"]
                assert update["fields"] == "userEnteredValue"
                for r, row in enumerate(update["rows"], start=start["rowIndex"]):
                    for c, cell in enumerate(row["values"], start=start["columnIndex"]):
                        value = cell.get("userEnteredValue", {})
                        assert len(value) <= 1
                        if "stringValue" in value:
                            assert isinstance(value["stringValue"], str)
                            tabs[gid][r][c] = value["stringValue"]
                        elif "boolValue" in value:
                            assert isinstance(value["boolValue"], bool)
                            tabs[gid][r][c] = value["boolValue"]
                        elif "numberValue" in value:
                            assert isinstance(value["numberValue"], (int, float))
                            assert not isinstance(value["numberValue"], bool)
                            tabs[gid][r][c] = value["numberValue"]
                        else:
                            assert not value  # Formula interpretation is never permitted.
                            tabs[gid][r][c] = ""
            elif "createDeveloperMetadata" in request:
                item = deepcopy(request["createDeveloperMetadata"]["developerMetadata"])
                gid = item["location"]["sheetId"]
                assert gid in tabs and item["visibility"] == "DOCUMENT"
                item["location"]["locationType"] = "SHEET"
                metadata.setdefault(gid, []).append(item)
            else:
                raise AssertionError(f"Unsupported Sheets operation: {request}")
        api.tabs, api.titles, api.metadata, api.grids = tabs, titles, metadata, grids
        # Keep the original raw list live so simulated edits remain visible.
        values[:] = tabs[42]
        api.tabs[42] = values
        return {"spreadsheetId": "demo_sheet", "replies": [{} for _ in body["requests"]]}

    api.spreadsheet.fetch_sheet_metadata.side_effect = metadata
    api.atomic_batch = batch_update
    api.spreadsheet.batch_update.side_effect = batch_update

    def read(gid):
        api.selected_gid = gid
        return deepcopy(api.tabs[gid])

    api.read = mocker.Mock(side_effect=read)

    def worksheet(gid):
        if gid not in api.tabs:
            raise gspread.WorksheetNotFound(str(gid))
        return SimpleNamespace(id=gid, title=api.titles.get(gid, f"Outcomes {gid}"), get=lambda **_: api.read(gid))

    def worksheet_by_title(title):
        for gid in api.tabs:
            if api.titles.get(gid) == title:
                return worksheet(gid)
        raise gspread.WorksheetNotFound(title)

    api.spreadsheet.get_worksheet_by_id.side_effect = worksheet
    api.spreadsheet.get_worksheet.side_effect = lambda index: worksheet(next(iter(api.tabs))) if index == 0 else None
    api.spreadsheet.worksheet.side_effect = worksheet_by_title
    api.client.open_by_key.return_value = api.spreadsheet
    api.service_account = mocker.patch(
        "xngin.apiserver.dwh.google_sheets.gspread.service_account", return_value=api.client
    )
    return api, values


def google_api_error(status_code):
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps({
        "error": {"code": status_code, "message": "PRIVATE-CREDENTIAL-CONTENT", "status": "ERROR"}
    }).encode()
    return gspread.exceptions.APIError(response)


def test_dsn_roundtrip_and_storage():
    api_dsn = aapi.GoogleSheetsDsn(spreadsheet_url=URL)
    config = RemoteDatabaseConfig(type="remote", dwh=GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB))
    reloaded = RemoteDatabaseConfig.model_validate_json(config.model_dump_json())
    assert isinstance(reloaded.dwh, GoogleSheetsDsn)
    assert reloaded.dwh.table_name == TAB
    assert settings_dwh_to_api_dsn(reloaded.dwh).model_dump() == {
        "type": "google_sheets",
        "spreadsheet_url": URL,
    }
    assert settings_dwh_to_api_dsn(reloaded.dwh) == api_dsn
    with pytest.raises(ValueError):
        aapi.GoogleSheetsDsn.model_validate({"spreadsheet_url": URL, "table_name": TAB})


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.google.com/spreadsheets/d/demo",
        "https://evil.com/spreadsheets/d/demo",
        "https://docs.google.com.evil.com/spreadsheets/d/demo",
        "https://docs.google.com/spreadsheets/d/demo#gid=nope",
        "https://docs.google.com/spreadsheets/d/demo#gid=",
        "https://docs.google.com/spreadsheets/d/demo?gid=",
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


def test_queries_use_typed_cells_and_refresh_each_session(sheets_api):
    api, values = sheets_api
    metric = DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB)) as dwh:
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
        values[3][2] = "50"  # An in-flight Sheets edit must not change this session's snapshot.
        outcomes = dwh.run(get_participant_metrics, table, [metric], "participant_id", ["002"])
        assert outcomes[0].metric_values[0].metric_value is None
        assert api.read.call_count == 1  # All queries in a request use one consistent snapshot.
    # A manual edit becomes visible on the next analysis request, without a schema-cache refresh.
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB)) as dwh:
        table = dwh.inspect_table(TAB)
        stats = dwh.run(get_raw_metric_stats, table, [metric], [])
        assert stats["outcome__count"] == 3
        assert stats["outcome__mean"] == 30
        assert stats["outcome__stddev"] == pytest.approx(np.std([10, 30, 50]))
    assert api.read.call_count == 2


@pytest.mark.parametrize("connectivity_check", [False, True])
@pytest.mark.parametrize(
    ("spreadsheet_title", "worksheet_title"),
    [
        ("My spreadsheet", "Raw data"),
        ("Café", "Raw data"),
        ("Café / Raw?#%&+", "data"),
        ("n" * (MAX_LENGTH_OF_NAME_VALUE - len(" - Data")), "Data"),
    ],
)
def test_creation_captures_google_name_as_the_persisted_table(
    sheets_api, aclient, xngin_session, spreadsheet_title, worksheet_title, connectivity_check
):
    api, _ = sheets_api
    api.spreadsheet.title = spreadsheet_title
    api.titles[42] = worksheet_title
    table_name = f"{spreadsheet_title} - {worksheet_title}"
    organization = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="Named sheet")).data
    datasource_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=organization.id, name="Connector", dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL)
        ),
        connectivity_check=connectivity_check,
    ).data.id

    xngin_session.expire_all()
    datasource = xngin_session.get_one(tables.Datasource, datasource_id)
    assert datasource.config["dwh"]["table_name"] == table_name
    assert datasource.table_list == [table_name]
    assert datasource.table_list_updated is not None
    assert api.read.call_count == 1
    assert aclient.inspect_datasource(datasource_id=datasource_id).data.tables == [table_name]
    assert api.read.call_count == 1
    # Exercise the same single encoded path segment used by browser callers, including decoded slashes.
    inspection_url = f"/v1/m/datasources/{datasource_id}/inspect/{quote(table_name, safe='')}"
    inspection = aclient.client.get(inspection_url)
    assert inspection.status_code == 200
    assert "participant_id" in inspection.json()["detected_unique_id_fields"]
    assert "display_name" not in inspection.json()
    assert api.read.call_count == 2
    cached = xngin_session.get_one(tables.DatasourceTablesInspected, (datasource_id, table_name))
    assert cached.response == inspection.json()
    # An ordinary fresh inspection cache remains usable without display-name-specific validation.
    api.spreadsheet.title = "A later Google rename"
    assert aclient.client.get(inspection_url).json() == inspection.json()
    assert api.read.call_count == 2
    refreshed = aclient.client.get(inspection_url, params={"refresh": True})
    assert refreshed.status_code == 200
    assert refreshed.json() == inspection.json()
    assert api.read.call_count == 3
    assert aclient.inspect_datasource(datasource_id=datasource_id, refresh=True).data.tables == [table_name]
    assert api.read.call_count == 3
    assert datasource.config["dwh"]["table_name"] == table_name


def test_captured_name_rejects_names_that_exceed_the_table_name_limit(sheets_api):
    api, _ = sheets_api
    api.spreadsheet.title = "n" * (MAX_LENGTH_OF_NAME_VALUE - len(" - Data") + 1)
    with (
        pytest.raises(LateValidationError, match=f"at most {MAX_LENGTH_OF_NAME_VALUE} characters"),
        DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh,
    ):
        dwh.list_tables()


@pytest.mark.parametrize("later_title", ["Renamed spreadsheet", "n" * (MAX_LENGTH_OF_NAME_VALUE + 1)])
def test_saved_connection_does_not_rediscover_its_name(sheets_api, later_title):
    api, values = sheets_api
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as discovery:
        captured_name = discovery.list_tables()[0]
    config = GoogleSheetsDsn(spreadsheet_url=URL, table_name=captured_name)
    api.spreadsheet.title = later_title
    api.titles[42] = "Different tab"
    values[1][2] = 50
    with DwhSession.open(config) as dwh:
        assert dwh.list_tables() == [TAB]
        table = dwh.inspect_table(TAB)
        assert table.name == TAB
        stats = dwh.run(
            get_raw_metric_stats, table, [DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)], []
        )
        assert stats["outcome__mean"] == 40
        assert dwh.inspect_table("42").name == "42"
        assert dwh.inspect_table("Different tab").name == "Different tab"
        assert api.read.call_count == 2  # One read during discovery, one for all three aliases in this session.


@pytest.mark.parametrize("table_reference", ["99", "Outcomes", "Evidential Demo Sheet - Outcomes"])
def test_tables_are_selected_within_the_spreadsheet_without_reading_the_source(sheets_api, table_reference):
    api, _ = sheets_api
    api.tabs[99] = [["participant_id", "outcome"], ["001", "7"]]
    api.titles[99] = "Outcomes" if table_reference == "99" else table_reference
    del api.tabs[42], api.titles[42]
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB)) as dwh:
        table = dwh.inspect_table(table_reference)
        outcomes = dwh.run(
            get_participant_metrics,
            table,
            [DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)],
            "participant_id",
            ["001"],
        )
    assert outcomes[0].metric_values[0].metric_value == 7
    api.read.assert_called_once_with(99)


@pytest.mark.parametrize("cached_tables", [None, ["Stale table name"]])
@pytest.mark.parametrize("refresh", [False, True])
def test_saved_table_list_needs_no_google_download_even_with_a_stale_cache(
    sheets_api, aclient, xngin_session, refresh, cached_tables
):
    api, _ = sheets_api
    organization = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="Offline sheet")).data
    datasource_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=organization.id, name="Connector", dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL)
        ),
    ).data.id
    datasource = xngin_session.get_one(tables.Datasource, datasource_id)
    datasource.table_list = cached_tables
    stale_updated = datetime.now(UTC) - timedelta(days=30)
    datasource.table_list_updated = stale_updated
    xngin_session.commit()
    reads = api.read.call_count
    api.read.side_effect = AssertionError("Listing a captured name must not contact Google")
    assert aclient.inspect_datasource(datasource_id=datasource_id, refresh=refresh).data.tables == [TAB]
    assert api.read.call_count == reads
    xngin_session.expire_all()
    assert datasource.table_list == cached_tables
    assert datasource.table_list_updated == stale_updated


def test_datasource_updates_preserve_the_saved_name_and_bind_the_new_url(sheets_api, aclient, xngin_session):
    api, _ = sheets_api
    organization = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="Editable sheet")).data
    datasource_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=organization.id, name="Connector", dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL)
        ),
    ).data.id
    api.tabs[99] = [["participant_id", "outcome"], ["001", "900"], ["002", "100"]]
    api.titles[99] = "A different tab must not replace the saved name"
    reads = api.read.call_count
    aclient.update_datasource(datasource_id=datasource_id, body=aapi.UpdateDatasourceRequest(name="Renamed connector"))
    new_url = URL.replace("gid=42", "gid=99")
    aclient.update_datasource(
        datasource_id=datasource_id,
        body=aapi.UpdateDatasourceRequest(dsn=aapi.GoogleSheetsDsn(spreadsheet_url=new_url)),
    )
    assert api.read.call_count == reads
    xngin_session.expire_all()
    datasource = xngin_session.get_one(tables.Datasource, datasource_id)
    config = datasource.get_config().dwh
    assert isinstance(config, GoogleSheetsDsn)
    assert config.table_name == TAB
    assert config.spreadsheet_url == new_url
    assert datasource.table_list == [TAB]
    with DwhSession.open(config) as dwh:
        stats = dwh.run(
            get_raw_metric_stats,
            dwh.inspect_table(TAB),
            [DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)],
            [],
        )
    assert stats["outcome__mean"] == 500
    assert api.selected_gid == 99


def test_switching_to_sheets_captures_a_name(sheets_api, aclient, xngin_session):
    organization = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="Switch connector")).data
    datasource_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(organization_id=organization.id, name="Connector", dsn=aapi.ApiOnlyDsn())
    ).data.id
    aclient.update_datasource(
        datasource_id=datasource_id, body=aapi.UpdateDatasourceRequest(dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL))
    )
    xngin_session.expire_all()
    datasource = xngin_session.get_one(tables.Datasource, datasource_id)
    assert datasource.config["dwh"]["table_name"] == TAB
    assert datasource.table_list == [TAB]


def test_empty_metric_is_numeric_and_zero_is_observed(sheets_api):
    _, values = sheets_api
    values[:] = [["id", "outcome"], ["001", ""], ["002", 0]]
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB)) as dwh:
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
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB)) as dwh:
        assert isinstance(dwh.inspect_table(TAB).c.outcome.type, sa.Double)


def test_assignments_match_ids_after_reordering_and_preserve_other_columns(sheets_api):
    _, values = sheets_api
    values[1], values[4] = values[4], values[1]
    data = SheetData.from_values(values).assignments_values(
        "participant_id", "demo-123", {"001": "Treatment", "003": "Control"}
    )
    assert data == [
        ["participant_id", "region", "outcome", "converted", "evidential_demo_123_arm"],
        ["003", "north", 30, True, "Control"],
        [None, None, None, None, None],
        ["002", "south", None, False, None],
        ["001", "north", 10, True, "Treatment"],
    ]
    assert values[0] == ["participant_id", "region", "outcome", "converted"]


def test_assignments_match_numeric_ids_as_stored_without_changing_source_cells(sheets_api):
    _, values = sheets_api
    values[:] = [["reference", "outcome"], ["+1", "10"], [" 2 ", "20"], ["3", "30"]]
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB)) as dwh:
        participants = dwh.get_participants(TAB, select_columns={"reference"}, n=3, filters=[]).participants
        assignments = {str(row.reference): "Treatment" for row in participants}
    data = SheetData.from_values(values).assignments_values("reference", "demo", assignments)
    assert data == [
        ["reference", "outcome", "evidential_demo_arm"],
        ["+1", 10, "Treatment"],
        [" 2 ", 20, "Treatment"],
        ["3", 30, "Treatment"],
    ]


def test_assignments_reuse_their_column(sheets_api):
    _, values = sheets_api
    values[0].append("evidential_demo_arm")
    values[1].append("Old treatment name")
    data = SheetData.from_values(values).assignments_values("participant_id", "demo", {"001": "Treatment"})
    assert data[0].count("evidential_demo_arm") == 1
    assert data[1][-1] == "Treatment"
    assert values[1][-1] == "Old treatment name"


def test_experiment_tab_keeps_selected_columns_and_matches_original_ids(sheets_api):
    _, values = sheets_api
    values[:] = [
        ["notes", "outcome", "participant_id", "unused_metric", "evidential_other_arm"],
        ["internal", "", "001", 20, "Old control"],
        [],
        ["internal", 0, "002", 30, "Old treatment"],
        ["internal", 5, "003", 40, "Old control"],
    ]
    data = SheetData.from_values(values).assignments_values(
        "participant_id", "demo", {"001": "Treatment", "003": "Control"}, field_names={"outcome"}
    )
    assert data == [
        ["outcome", "participant_id", "evidential_demo_arm"],
        [None, "001", "Treatment"],
        [None, None, None],
        [0, "002", None],
        [5, "003", "Control"],
    ]
    assert values[0][0] == "notes"


def test_experiment_tab_rejects_missing_selected_columns(sheets_api):
    _, values = sheets_api
    with pytest.raises(LateValidationError, match=r"Experiment columns are missing.*missing_metric"):
        SheetData.from_values(values).assignments_values("participant_id", "demo", {}, field_names={"missing_metric"})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "unique"),
        ("missing", "missing"),
        ("blank_id", "participant ID"),
        ("missing_column", "column is missing"),
    ],
)
def test_experiment_tab_rejects_invalid_participant_ids(sheets_api, mutation, match):
    _, values = sheets_api
    if mutation == "duplicate":
        values[4][0] = "001"
    elif mutation == "missing":
        values[1][0] = "new-id"
    elif mutation == "blank_id":
        values[4][0] = ""
    else:
        values[0][0] = "other_id"
    with pytest.raises(LateValidationError, match=match):
        SheetData.from_values(values).assignments_values("participant_id", "demo", {"001": "Treatment"})


def test_sheet_limits_reject_data_instead_of_truncating():
    with pytest.raises(LateValidationError, match="5,000"):
        SheetData.from_values([["id"], *[[str(i)] for i in range(5_001)]])
    with pytest.raises(LateValidationError, match="100"):
        SheetData.from_values([[f"col_{i}" for i in range(101)]])


def test_sheet_limits_accept_the_full_supported_grid():
    values = [[f"col_{i}" for i in range(100)], *[[str(i)] for i in range(5_000)]]
    data = SheetData.from_values(values)
    assert len(data.headers) == 100
    assert len(data.rows) == 5_000
    assert data.rows[-1][0] == 5_001
    assert data.rows[-1][1] == ["4999", *[None] * 99]


@pytest.fixture
def sheets_transport(mocker):
    sessions = []

    def make_response(status_code, body, headers=None):
        response = requests.Response()
        response.status_code = status_code
        response.headers.update(headers or {})
        response.url = "https://sheets.googleapis.com/v4/spreadsheets/demo_sheet"
        response.raw = io.BytesIO(body)
        # Keep requests' real redirect handling and Response streaming/JSON decoding.
        adapter = mocker.Mock(spec=requests.adapters.BaseAdapter)
        adapter.send.return_value = response
        session = requests.Session()
        session.trust_env = False
        session.mount("https://", adapter)
        sessions.append(session)
        return SheetsHTTPClient(AnonymousCredentials(), session=session), response.url

    yield make_response
    for session in sessions:
        session.close()


@pytest.mark.parametrize("location", ["https://sheets.googleapis.com/elsewhere", "https://evil.example/private"])
def test_api_response_redirect_is_rejected(sheets_transport, location):
    transport, url = sheets_transport(302, b"", {"Location": location})
    with pytest.raises(DwhConnectionError, match="redirect"):
        transport.request("get", url)


@pytest.mark.parametrize("declared_length", [False, True])
def test_api_response_size_limit_is_enforced_before_parsing(sheets_transport, declared_length):
    headers = {"Content-Length": str(MAX_SHEET_BYTES + 1)} if declared_length else {}
    transport, url = sheets_transport(200, b"[" + b" " * MAX_SHEET_BYTES + b"]", headers)
    with pytest.raises(LateValidationError, match="8"):
        transport.request("get", url)


def test_api_response_at_byte_limit_is_available_for_json_decoding(sheets_transport):
    body = json.dumps({"values": [["id"], ["001"]], "padding": ""}).encode()
    body = body[:-2] + b"x" * (MAX_SHEET_BYTES - len(body)) + body[-2:]
    transport, url = sheets_transport(200, body)
    response = transport.request("get", url)
    assert response.json()["values"] == [["id"], ["001"]]
    assert len(response.content) == MAX_SHEET_BYTES


def test_api_error_response_keeps_status_for_actionable_connection_errors(sheets_transport):
    body = b'{"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}'
    transport, url = sheets_transport(429, body)
    with pytest.raises(gspread.exceptions.APIError) as error:
        transport.request("get", url)
    assert error.value.response.status_code == 429


@pytest.mark.parametrize("configured_path", [None, ""])
def test_missing_credentials_explain_required_configuration(mocker, configured_path):
    mocker.patch.object(flags, "GOOGLE_SHEETS_CREDENTIALS_FILE", configured_path)
    with pytest.raises(DwhConnectionError, match="XNGIN_GOOGLE_SHEETS_CREDENTIALS_FILE"):
        GoogleSheetsClient(URL)


@pytest.mark.parametrize("keyfile", ["missing", "malformed", "invalid_private_key"])
def test_invalid_keyfile_reports_configuration_without_exposing_contents(mocker, tmp_path, keyfile):
    path = tmp_path / "service-account.json"
    secret = "PRIVATE-CREDENTIAL-CONTENT-MUST-NOT-LEAK"
    if keyfile == "malformed":
        path.write_text(f"not-json {secret}")
    elif keyfile == "invalid_private_key":
        path.write_text(
            json.dumps({
                "type": "service_account",
                "client_email": "reader@example.iam.gserviceaccount.com",
                "token_uri": "https://oauth2.googleapis.com/token",
                "private_key": secret,
            })
        )
    mocker.patch.object(flags, "GOOGLE_SHEETS_CREDENTIALS_FILE", str(path))
    with pytest.raises(DwhConnectionError, match="XNGIN_GOOGLE_SHEETS_CREDENTIALS_FILE") as error:
        GoogleSheetsClient(URL)
    assert secret not in str(error.value)


@pytest.mark.parametrize("error_type", [PermissionError, gspread.SpreadsheetNotFound])
def test_inaccessible_spreadsheet_explains_service_account_sharing(sheets_api, error_type):
    api, _ = sheets_api
    api.client.open_by_key.side_effect = error_type("PRIVATE-CREDENTIAL-CONTENT")
    with pytest.raises(DwhConnectionError) as error:
        GoogleSheetsClient(URL)
    assert api.client.http_client.auth.service_account_email in str(error.value)
    assert "PRIVATE-CREDENTIAL-CONTENT" not in str(error.value)


def test_datasource_access_error_identifies_the_configured_account(sheets_api, aclient):
    api, _ = sheets_api
    email = "different-reader@another-project.iam.gserviceaccount.com"
    api.client.http_client.auth.service_account_email = email
    api.client.open_by_key.side_effect = PermissionError("PRIVATE-CREDENTIAL-CONTENT")
    organization = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="Sharing error")).data
    response = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=organization.id,
            name="Private sheet",
            dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL),
        ),
        connectivity_check=True,
        raise_if_not_default_status=False,
    )
    assert response.status == 502
    assert email in response.data.message
    assert "CONNECTION ERROR" not in response.data.message
    assert "PRIVATE-CREDENTIAL-CONTENT" not in response.data.message


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 429, 500])
@pytest.mark.parametrize("operation", ["open", "read"])
def test_api_errors_are_actionable_without_leaking_remote_content(sheets_api, status_code, operation):
    api, _ = sheets_api
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps({
        "error": {"code": status_code, "message": "PRIVATE-CREDENTIAL-CONTENT", "status": "ERROR"}
    }).encode()
    failure = gspread.exceptions.APIError(response)
    if operation == "open":
        api.client.open_by_key.side_effect = failure
    else:
        api.read.side_effect = failure
    expected = (
        api.client.http_client.auth.service_account_email
        if status_code in {401, 403, 404}
        else "rate limit"
        if status_code == 429
        else f"HTTP {status_code}"
    )
    with (
        pytest.raises(DwhConnectionError, match=expected) as error,
        DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh,
    ):
        dwh.list_tables()
    assert "PRIVATE-CREDENTIAL-CONTENT" not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [RefreshError("PRIVATE-CREDENTIAL-CONTENT"), requests.Timeout("PRIVATE-CREDENTIAL-CONTENT")],
)
def test_authentication_and_transport_errors_do_not_expose_credentials(sheets_api, failure):
    api, _ = sheets_api
    api.read.side_effect = failure
    with (
        pytest.raises(DwhConnectionError, match="Google Sheets") as error,
        DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL)) as dwh,
    ):
        dwh.list_tables()
    assert "PRIVATE-CREDENTIAL-CONTENT" not in str(error.value)


def test_missing_worksheet_explains_how_to_reconnect_the_intended_tab(sheets_api):
    with (
        pytest.raises(DwhConnectionError, match="selected tab"),
        DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL.replace("gid=42", "gid=999"))) as dwh,
    ):
        dwh.list_tables()


def test_url_without_gid_uses_first_tab_and_zero_selects_its_own_tab(sheets_api):
    api, _ = sheets_api
    api.tabs[0] = [["participant_id", "outcome"], ["different", "900"]]
    api.titles[0] = "Zero tab"
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL.split("#", maxsplit=1)[0])) as dwh:
        assert dwh.list_tables() == [TAB]
        participants = dwh.get_participants(TAB, select_columns={"participant_id"}, n=10, filters=[]).participants
        assert {row.participant_id for row in participants} == {"001", "002", "003"}
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL.replace("gid=42", "gid=0"))) as dwh:
        name = "Evidential Demo Sheet - Zero tab"
        participants = dwh.get_participants(name, select_columns={"participant_id"}, n=10, filters=[]).participants
        assert [row.participant_id for row in participants] == ["different"]


def test_formatted_cells_preserve_missing_zero_false_and_textual_ids(sheets_api):
    _, values = sheets_api
    values[:] = [
        ["participant_id", "outcome", "converted"],
        ["001", "", ""],
        ["002", "0", "FALSE"],
        ["003", "1", "TRUE"],
    ]
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB)) as dwh:
        participants = dwh.get_participants(
            TAB, select_columns={"participant_id", "outcome", "converted"}, n=10, filters=[]
        ).participants
    assert {row.participant_id: (row.outcome, row.converted) for row in participants} == {
        "001": (None, None),
        "002": (0, False),
        "003": (1, True),
    }


@pytest.mark.parametrize("blank_outcomes", [False, True])
@pytest.mark.parametrize("cluster_key", [None, "region"])
def test_create_launch_and_analyze_a_sheet_experiment(sheets_api, aclient, xngin_session, blank_outcomes, cluster_key):
    api, values = sheets_api
    examples = Path(__file__).parents[4] / "tools"
    with (examples / "google_sheets_example.csv").open(newline="") as source:
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
    reads_before_launch = api.read.call_count
    aclient.commit_experiment(datasource_id=datasource_id, experiment_id=experiment_id)
    assert api.read.call_count > reads_before_launch
    xngin_session.expire_all()
    experiment = xngin_session.get_one(tables.Experiment, experiment_id)
    assert experiment.state == ExperimentState.COMMITTED
    assert experiment.datasource_table is not None
    outcome_gid = int(experiment.datasource_table)
    assert outcome_gid != 42
    config = aclient.get_experiment_for_ui(datasource_id=datasource_id, experiment_id=experiment_id).data.config
    assert (
        config.google_sheets_experiment_url
        == f"https://docs.google.com/spreadsheets/d/demo_sheet/edit#gid={outcome_gid}"
    )
    assert config.design_spec.table_name == str(outcome_gid)
    assert values[0] == headers
    expected_fields = {"participant_id", *(metric.field_name for metric in metrics)}
    if cluster_key is not None:
        expected_fields.add(cluster_key)
    assert set(api.tabs[outcome_gid][0][:-1]) == expected_fields
    # Once launched, analysis and recommitting must work without the source tab.
    del api.tabs[42], api.titles[42]
    values = api.tabs[outcome_gid]
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
    # Observations in only one arm must not produce a fabricated control comparison.
    onboarding_index = values[0].index("onboarded_within_1_week")
    for row in values[1:]:
        row[onboarding_index] = "1" if row[-1] == "Treatment" else ""
    partial = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert isinstance(partial, FreqExperimentAnalysisResponse)
    onboarding_partial = next(m for m in partial.metric_analyses if m.metric_name == "onboarded_within_1_week")
    assert all(arm.num_missing_values == -1 for arm in onboarding_partial.arm_analyses)
    for row in values[1:]:
        row[onboarding_index] = "0" if row[-1] == "Control" else ""
    partial = aclient.analyze_experiment(datasource_id=datasource_id, experiment_id=experiment_id).data
    assert isinstance(partial, FreqExperimentAnalysisResponse)
    onboarding_partial = next(m for m in partial.metric_analyses if m.metric_name == "onboarded_within_1_week")
    control = next(arm for arm in onboarding_partial.arm_analyses if arm.is_baseline)
    treatment = next(arm for arm in onboarding_partial.arm_analyses if not arm.is_baseline)
    assert control.num_missing_values == 0
    assert control.estimate == pytest.approx(0)
    assert treatment.num_missing_values == -1
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
    after_by_metric = {metric.metric_name: metric for metric in after.metric_analyses}
    for previous in before.metric_analyses:
        if previous.metric_name == "minutes_on_site_last_7_days":
            continue
        current = after_by_metric[previous.metric_name]
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
    assert 42 not in api.tabs
    assert api.selected_gid == outcome_gid


@pytest.mark.parametrize(
    ("spreadsheet_title", "worksheet_title"),
    [("Evidential Demo Sheet", "Data"), ("Café / Raw?#%&+", "data")],
)
def test_shared_raw_sheet_has_isolated_experiment_outcomes(
    sheets_api, aclient, xngin_session, spreadsheet_title, worksheet_title
):
    from xngin.apiserver.snapshots.snapshotter import (  # noqa: PLC0415
        create_pending_snapshots,
        process_pending_snapshots,
    )

    api, raw = sheets_api
    api.spreadsheet.title = spreadsheet_title
    api.titles[42] = worksheet_title
    table_name = f"{spreadsheet_title} - {worksheet_title}"
    raw[:] = [["participant_id", "outcome"], *[[f"{i:03}", str(i)] for i in range(1, 21)]]
    original_raw = [row.copy() for row in raw]
    org_id = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="Shared sheet")).data.id
    ds_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=org_id, name="Raw participants", dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL)
        ),
        connectivity_check=True,
    ).data.id
    now = datetime.now(UTC)

    def create(name):
        response = aclient.create_experiment(
            datasource_id=ds_id,
            random_state=42,
            body=CreateExperimentRequest(
                design_spec=PreassignedFrequentistExperimentSpec(
                    experiment_type="freq_preassigned",
                    experiment_name=name,
                    description="Shared raw source, independent outcomes",
                    table_name=table_name,
                    primary_key="participant_id",
                    start_date=now - timedelta(days=1),
                    end_date=now + timedelta(days=1),
                    desired_n=20,
                    metrics=[DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)],
                    strata=[],
                    filters=[],
                    arms=[
                        Arm(arm_name="Control", arm_description="Control"),
                        Arm(arm_name="Treatment", arm_description="Treatment"),
                    ],
                )
            ),
        ).data
        assert response.google_sheets_experiment_url is None
        aclient.commit_experiment(datasource_id=ds_id, experiment_id=response.experiment_id)
        return response.experiment_id

    first, second = create("First"), create("Second")
    experiment_tables = {
        xngin_session.get_one(tables.Experiment, experiment_id).datasource_table for experiment_id in (first, second)
    }
    assert len(experiment_tables) == 2 and table_name not in experiment_tables
    # Changing the default source connection must not change either experiment's tab ID.
    api.spreadsheet.title = "Renamed raw spreadsheet"
    reads = api.read.call_count
    aclient.update_datasource(
        datasource_id=ds_id, body=aapi.UpdateDatasourceRequest(dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL))
    )
    assert api.read.call_count == reads

    initial_assignments = list(
        xngin_session.execute(
            sa
            .select(
                tables.ArmAssignment.experiment_id, tables.ArmAssignment.participant_id, tables.ArmAssignment.arm_id
            )
            .where(tables.ArmAssignment.experiment_id.in_([first, second]))
            .order_by(tables.ArmAssignment.experiment_id, tables.ArmAssignment.participant_id)
        )
    )
    gids = {}
    for experiment_id, effect in [(first, 100), (second, 200)]:
        xngin_session.expire_all()
        experiment_table = xngin_session.get_one(tables.Experiment, experiment_id).datasource_table
        assert experiment_table is not None
        gid = int(experiment_table)
        config = aclient.get_experiment_for_ui(datasource_id=ds_id, experiment_id=experiment_id).data.config
        assert (
            config.google_sheets_experiment_url == f"https://docs.google.com/spreadsheets/d/demo_sheet/edit#gid={gid}"
        )
        gids[experiment_id] = gid
        assert gid != 42
        tab = api.tabs[gid]
        for i, row in enumerate(tab[1:]):
            row[1] = str(i + (effect if row[-1] == "Treatment" else 0))
    assert gids[first] != gids[second]

    def treatment(experiment_id):
        analysis = aclient.analyze_experiment(datasource_id=ds_id, experiment_id=experiment_id).data
        assert isinstance(analysis, FreqExperimentAnalysisResponse)
        return next(arm.estimate for arm in analysis.metric_analyses[0].arm_analyses if not arm.is_baseline)

    first_result, second_result = treatment(first), treatment(second)
    assert second_result == pytest.approx(first_result + 100)
    create_pending_snapshots(3600)
    process_pending_snapshots(90, max_jitter_secs=0)
    xngin_session.expire_all()
    saved_snapshots = list(
        xngin_session.scalars(sa.select(tables.Snapshot).where(tables.Snapshot.experiment_id.in_([first, second])))
    )
    assert {snapshot.experiment_id for snapshot in saved_snapshots} == {first, second}
    for snapshot in saved_snapshots:
        assert snapshot.status == "success", snapshot.message
        analysis = FreqExperimentAnalysisResponse.model_validate(snapshot.data)
        estimate = next(arm.estimate for arm in analysis.metric_analyses[0].arm_analyses if not arm.is_baseline)
        assert estimate == pytest.approx(first_result if snapshot.experiment_id == first else second_result)
    saved_data = {(snapshot.experiment_id, snapshot.id): snapshot.data for snapshot in saved_snapshots}
    aclient.update_experiment(
        datasource_id=ds_id, experiment_id=first, body=aapi.UpdateExperimentRequest(name="Renamed")
    )
    assert treatment(first) == pytest.approx(first_result)
    # Recommit and tab renames never replace observations or repoint another experiment.
    api.titles[gids[first]] = "Renamed outcomes"
    outcomes_before = deepcopy(api.tabs)
    batches_before = api.spreadsheet.batch_update.call_count
    reads_before = api.read.call_count
    aclient.commit_experiment(datasource_id=ds_id, experiment_id=first)
    assert api.read.call_count == reads_before
    assert api.spreadsheet.batch_update.call_count == batches_before
    assert api.tabs == outcomes_before
    assert treatment(first) == pytest.approx(first_result)
    assert treatment(second) == pytest.approx(second_result)
    assert raw == original_raw
    datasource_config = xngin_session.get_one(tables.Datasource, ds_id).get_config().dwh
    assert isinstance(datasource_config, GoogleSheetsDsn)
    assert datasource_config.spreadsheet_url == URL
    assert datasource_config.table_name == table_name
    assert {
        xngin_session.get_one(tables.Experiment, experiment_id).datasource_table for experiment_id in (first, second)
    } == experiment_tables
    assert aclient.inspect_datasource(datasource_id=ds_id).data.tables == [table_name]
    with DwhSession.open(xngin_session.get_one(tables.Datasource, ds_id).get_config().dwh) as dwh:
        stats = dwh.run(
            get_raw_metric_stats,
            dwh.inspect_table(table_name),
            [DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)],
            [],
        )
    assert stats["outcome__count"] == 20
    assert stats["outcome__mean"] == pytest.approx(10.5)
    assert initial_assignments == list(
        xngin_session.execute(
            sa
            .select(
                tables.ArmAssignment.experiment_id, tables.ArmAssignment.participant_id, tables.ArmAssignment.arm_id
            )
            .where(tables.ArmAssignment.experiment_id.in_([first, second]))
            .order_by(tables.ArmAssignment.experiment_id, tables.ArmAssignment.participant_id)
        )
    )
    xngin_session.expire_all()
    for snapshot_key, data in saved_data.items():
        assert xngin_session.get_one(tables.Snapshot, snapshot_key).data == data


def test_atomic_writeback_preserves_cells_selected_columns_and_raw_tab(sheets_api, mocker):
    api, raw = sheets_api
    raw[:] = [
        ["notes", "participant_id", "outcome", "converted", "unused_metric"],
        ["omit", "001", "", False, 19],
        [],
        ["omit", "=2+2", 0, "FALSE", 23],
        ["omit", "9007199254740993", "=SUM(A1:A2)", True, 29],
        ["omit", "+1", "03", "", 31],
    ]
    original = deepcopy(raw)
    mocker.patch("xngin.apiserver.dwh.google_sheets.secrets.randbelow", side_effect=[42, 0])
    with DwhSession.open(GoogleSheetsDsn(spreadsheet_url=URL, table_name=TAB)) as dwh:
        tab_id = dwh.create_sheet_experiment_tab(
            TAB,
            "participant_id",
            "demo",
            {"001": "=Treatment", "=2+2": "Control", "9007199254740993": "Control"},
            field_names={"outcome", "converted"},
        )
    assert tab_id == "0"
    assert raw == original and api.tabs[42] == original
    assert api.tabs[0] == [
        ["participant_id", "outcome", "converted", "evidential_demo_arm"],
        ["001", "", False, "=Treatment"],
        ["", "", "", ""],
        ["=2+2", 0, False, "Control"],
        ["9007199254740993", "=SUM(A1:A2)", True, "Control"],
        ["+1", "03", "", ""],
    ]
    assert api.grids[0] == {"rowCount": 6, "columnCount": 4}
    assert api.metadata[0] == [
        {
            "metadataKey": "evidential_experiment_id",
            "metadataValue": "demo",
            "location": {"sheetId": 0, "locationType": "SHEET"},
            "visibility": "DOCUMENT",
        }
    ]
    api.spreadsheet.batch_update.assert_called_once()
    api.writer.http_client.session.close.assert_called_once()


def test_writeback_uses_native_metric_types_without_converting_identifiers(sheets_api):
    api, raw = sheets_api
    raw[:] = [
        ["reference", "cohort_id", "mean_task_time_mins", "converted"],
        ["001", "0007", "5", "TRUE"],
        [],
        ["002", "0008", "6.5", "FALSE"],
        ["003", "0009", "0", ""],
        ["004", "0010", "", "TRUE"],
    ]
    original = deepcopy(raw)
    client = GoogleSheetsClient(URL, TAB)
    try:
        tab_id = client.create_experiment_tab(TAB, "reference", "native", {"001": "Treatment", "002": "Control"})
    finally:
        client.close()
    gid = int(tab_id)
    assert api.tabs[gid] == [
        ["reference", "cohort_id", "mean_task_time_mins", "converted", "evidential_native_arm"],
        ["001", "0007", 5, True, "Treatment"],
        ["", "", "", "", ""],
        ["002", "0008", 6.5, False, "Control"],
        ["003", "0009", 0, "", ""],
        ["004", "0010", "", True, ""],
    ]
    assert isinstance(api.tabs[gid][1][2], (int, float))
    assert isinstance(api.tabs[gid][1][3], bool)
    assert raw == original


def test_owned_tab_recovery_never_reads_a_deleted_source_tab(sheets_api):
    api, _ = sheets_api
    client = GoogleSheetsClient(URL, TAB)
    try:
        tab_id = client.create_experiment_tab(TAB, "participant_id", "demo", {"001": "Control"})
    finally:
        client.close()
    gid = int(tab_id)
    api.tabs[gid][1][2] = "1234"
    api.titles[gid] = "Renamed by the spreadsheet owner"
    outcomes = deepcopy(api.tabs[gid])
    del api.tabs[42], api.titles[42]
    api.read.reset_mock()
    client = GoogleSheetsClient(URL, TAB)
    try:
        recovered = client.create_experiment_tab(
            TAB, "participant_id", "demo", {"001": "New treatment"}, field_names={"deleted_column"}
        )
    finally:
        client.close()
    assert recovered == tab_id
    assert api.tabs[gid] == outcomes
    assert len(api.tabs) == 1
    api.read.assert_not_called()
    api.spreadsheet.batch_update.assert_called_once()


def test_writeback_never_adopts_a_title_or_row_metadata_without_sheet_ownership(sheets_api):
    api, raw = sheets_api
    api.tabs[99] = [["private"], ["leave alone"]]
    api.titles[99] = "Evidential demo"
    api.metadata[99] = [
        {
            "metadataKey": "evidential_experiment_id",
            "metadataValue": "demo",
            "location": {"dimensionRange": {"sheetId": 99, "dimension": "ROWS", "startIndex": 0, "endIndex": 1}},
        }
    ]
    original = deepcopy(api.tabs)
    client = GoogleSheetsClient(URL, TAB)
    try:
        tab_id = client.create_experiment_tab(TAB, "participant_id", "demo", {"001": "Control"})
    finally:
        client.close()
    gid = int(tab_id)
    assert gid not in original
    assert "demo" in api.titles[gid]
    assert api.titles[gid] != api.titles[99]
    assert api.tabs[99] == original[99] and raw == original[42]


@pytest.mark.parametrize("failure_at", [0, 1, 2])
def test_failed_atomic_batch_leaves_no_partial_tab_or_metadata(sheets_api, failure_at):
    api, raw = sheets_api
    original = deepcopy(raw)
    api.fail_request_at = failure_at
    client = GoogleSheetsClient(URL, TAB)
    try:
        with pytest.raises(DwhConnectionError, match="HTTP 400"):
            client.create_experiment_tab(TAB, "participant_id", "demo", {"001": "Control"})
    finally:
        client.close()
    assert api.tabs == {42: original}
    assert api.metadata == {} and api.grids == {}
    assert api.titles == {42: "Data"}
    api.writer.http_client.session.close.assert_called_once()


@pytest.mark.parametrize("operation", ["open", "metadata", "batch"])
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 429, 500])
def test_write_errors_are_actionable_and_do_not_expose_remote_content(sheets_api, operation, status_code):
    api, _ = sheets_api
    client = GoogleSheetsClient(URL, TAB)
    operation_mock = {
        "open": api.writer.open_by_key,
        "metadata": api.spreadsheet.fetch_sheet_metadata,
        "batch": api.spreadsheet.batch_update,
    }[operation]
    operation_mock.side_effect = google_api_error(status_code)
    try:
        with pytest.raises(DwhConnectionError) as error:
            client.create_experiment_tab(TAB, "participant_id", "demo", {"001": "Control"})
    finally:
        client.close()
    message = str(error.value)
    assert "PRIVATE-CREDENTIAL-CONTENT" not in message
    if status_code in {401, 403, 404}:
        assert api.client.http_client.auth.service_account_email in message
        assert "as Editor" in message and "share" in message.lower()
        assert "CONNECTION ERROR" not in message
    elif status_code == 429:
        assert "rate limit" in message
    else:
        assert f"HTTP {status_code}" in message
    assert set(api.tabs) == {42}
    api.writer.http_client.session.close.assert_called_once()


@pytest.mark.parametrize("error_type", [PermissionError, gspread.SpreadsheetNotFound, RefreshError, requests.Timeout])
def test_write_authentication_and_transport_failures_are_safe(sheets_api, error_type):
    api, _ = sheets_api
    client = GoogleSheetsClient(URL, TAB)
    api.writer.open_by_key.side_effect = error_type("PRIVATE-CREDENTIAL-CONTENT")
    try:
        with pytest.raises(DwhConnectionError) as error:
            client.create_experiment_tab(TAB, "participant_id", "demo", {"001": "Control"})
    finally:
        client.close()
    assert "PRIVATE-CREDENTIAL-CONTENT" not in str(error.value)
    if error_type in {PermissionError, gspread.SpreadsheetNotFound}:
        assert "as Editor" in str(error.value)
    assert set(api.tabs) == {42}


def test_assignment_column_is_unique_ignoring_case(sheets_api):
    api, raw = sheets_api
    raw[0].append("EVIDENTIAL_DEMO_ARM")
    raw[1].append("Outdated assignment")
    client = GoogleSheetsClient(URL, TAB)
    try:
        tab_id = client.create_experiment_tab(TAB, "participant_id", "demo", {"001": "Treatment"})
    finally:
        client.close()
    gid = int(tab_id)
    assert sum(name.lower() == "evidential_demo_arm" for name in api.tabs[gid][0]) == 1
    assert api.tabs[gid][1][-1] == "Treatment"
    assert raw[1][-1] == "Outdated assignment"


@pytest.fixture
def assigned_sheet_experiment(sheets_api, aclient):
    _, raw = sheets_api
    raw[:] = [["participant_id", "outcome"], *[[f"{i:03}", str(i)] for i in range(1, 21)]]
    organization_id = aclient.create_organizations(body=aapi.CreateOrganizationRequest(name="Writeback")).data.id
    datasource_id = aclient.create_datasource(
        body=aapi.CreateDatasourceRequest(
            organization_id=organization_id, name="Source", dsn=aapi.GoogleSheetsDsn(spreadsheet_url=URL)
        ),
    ).data.id
    now = datetime.now(UTC)
    experiment_id = aclient.create_experiment(
        datasource_id=datasource_id,
        random_state=42,
        body=CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_type="freq_preassigned",
                experiment_name="Writeback",
                description="Automatic outcomes",
                table_name=TAB,
                primary_key="participant_id",
                start_date=now - timedelta(days=1),
                end_date=now + timedelta(days=1),
                desired_n=20,
                metrics=[DesignSpecMetricRequest(field_name="outcome", metric_pct_change=0.1)],
                strata=[],
                filters=[],
                arms=[
                    Arm(arm_name="Control", arm_description="Control"),
                    Arm(arm_name="Treatment", arm_description="Treatment"),
                ],
            )
        ),
    ).data.experiment_id
    return SimpleNamespace(organization_id=organization_id, datasource_id=datasource_id, experiment_id=experiment_id)


@pytest.mark.parametrize("failure", ["editor", "quota", "timeout", "missing_id", "duplicate_id", "atomic_metadata"])
def test_commit_failures_retain_source_table_and_leave_no_events(
    sheets_api, assigned_sheet_experiment, aclient, xngin_session, failure
):
    api, raw = sheets_api
    setup = assigned_sheet_experiment
    original = deepcopy(raw)
    if failure == "editor":
        api.spreadsheet.batch_update.side_effect = google_api_error(403)
    elif failure == "quota":
        api.spreadsheet.batch_update.side_effect = google_api_error(429)
    elif failure == "timeout":
        api.spreadsheet.batch_update.side_effect = requests.Timeout("PRIVATE-CREDENTIAL-CONTENT")
    elif failure == "missing_id":
        raw[1][0] = "new-id"
    elif failure == "duplicate_id":
        raw[2][0] = raw[1][0]
    else:
        api.fail_request_at = 2
    events_before = list(xngin_session.scalars(sa.select(tables.Event.id)))
    tasks_before = list(xngin_session.scalars(sa.select(tables.Task.id)))
    response = aclient.commit_experiment(
        datasource_id=setup.datasource_id, experiment_id=setup.experiment_id, raise_if_not_default_status=False
    )
    assert response.status == (422 if failure in {"missing_id", "duplicate_id"} else 502)
    if failure == "editor":
        assert "as Editor" in response.data.message
        assert api.client.http_client.auth.service_account_email in response.data.message
        assert "CONNECTION ERROR" not in response.data.message
    xngin_session.expire_all()
    experiment = xngin_session.get_one(tables.Experiment, setup.experiment_id)
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_table == TAB
    assert list(xngin_session.scalars(sa.select(tables.Event.id))) == events_before
    assert list(xngin_session.scalars(sa.select(tables.Task.id))) == tasks_before
    assert set(api.tabs) == {42} and api.metadata == {}
    raw[:] = original
    if failure in {"editor", "quota", "timeout"}:
        api.spreadsheet.batch_update.side_effect = api.atomic_batch
    api.fail_request_at = None
    aclient.commit_experiment(datasource_id=setup.datasource_id, experiment_id=setup.experiment_id)
    xngin_session.expire_all()
    experiment = xngin_session.get_one(tables.Experiment, setup.experiment_id)
    assert experiment.state == ExperimentState.COMMITTED
    assert experiment.datasource_table is not None and experiment.datasource_table != TAB
    assert raw == original
    assert len(list(xngin_session.scalars(sa.select(tables.Event.id)))) == len(events_before) + 1


def test_remote_success_local_rollback_recovers_without_overwriting_outcomes(
    sheets_api, assigned_sheet_experiment, aclient, xngin_session
):
    from xngin.apiserver.routers.experiments.experiments_common import commit_experiment_impl  # noqa: PLC0415

    api, _ = sheets_api
    setup = assigned_sheet_experiment
    experiment = xngin_session.get_one(tables.Experiment, setup.experiment_id)
    events_before = list(xngin_session.scalars(sa.select(tables.Event.id)))
    transaction = xngin_session.begin_nested()
    commit_experiment_impl(xngin_session, experiment)
    xngin_session.flush()
    tab_id = experiment.datasource_table
    assert tab_id is not None
    gid = int(tab_id)
    transaction.rollback()
    xngin_session.expire_all()
    assert experiment.state == ExperimentState.ASSIGNED and experiment.datasource_table == TAB
    assert list(xngin_session.scalars(sa.select(tables.Event.id))) == events_before
    api.tabs[gid][1][1] = "9876"
    api.titles[gid] = "Recovered, renamed outcomes"
    outcomes = deepcopy(api.tabs[gid])
    del api.tabs[42], api.titles[42]
    api.read.reset_mock()
    aclient.commit_experiment(datasource_id=setup.datasource_id, experiment_id=setup.experiment_id)
    xngin_session.expire_all()
    assert experiment.datasource_table == tab_id
    assert experiment.state == ExperimentState.COMMITTED
    assert api.tabs[gid] == outcomes
    assert len(api.tabs) == 1
    api.read.assert_not_called()
    api.spreadsheet.batch_update.assert_called_once()
    assert len(list(xngin_session.scalars(sa.select(tables.Event.id)))) == len(events_before) + 1


@pytest.mark.parametrize("participant_value", [123, 123.0, "00123", "=1+2", "9007199254740993"])
def test_written_participant_ids_are_always_literal_strings(sheets_api, participant_value):
    api, raw = sheets_api
    raw[:] = [["participant_id", "outcome"], [participant_value, 0]]
    expected_id = "123" if participant_value == 123 else participant_value
    client = GoogleSheetsClient(URL, TAB)
    try:
        tab_id = client.create_experiment_tab(TAB, "participant_id", "demo", {expected_id: "Control"})
    finally:
        client.close()
    gid = int(tab_id)
    assert api.tabs[gid][1] == [expected_id, 0, "Control"]
    assert isinstance(api.tabs[gid][1][0], str)
    assert raw[1][0] == participant_value


def test_concurrent_commits_serialize_without_duplicate_tabs_or_events(
    sheets_api, assigned_sheet_experiment, xngin_session
):
    from xngin.apiserver.routers.experiments.experiments_common import commit_experiment_impl  # noqa: PLC0415

    api, _ = sheets_api
    setup = assigned_sheet_experiment
    engine = xngin_session.get_bind()
    ready = Barrier(2)
    events_before = set(xngin_session.scalars(sa.select(tables.Event.id)))

    def commit():
        with Session(engine) as session:
            experiment = session.get_one(tables.Experiment, setup.experiment_id)
            assert experiment.state == ExperimentState.ASSIGNED
            _ = experiment.datasource.config
            ready.wait(timeout=10)
            commit_experiment_impl(session, experiment)
            session.commit()
            return experiment.datasource_table

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(commit)
        second = executor.submit(commit)
        tab_ids = [first.result(timeout=20), second.result(timeout=20)]
    assert tab_ids[0] is not None and tab_ids[0] == tab_ids[1]
    assert len(api.tabs) == 2
    api.spreadsheet.batch_update.assert_called_once()
    xngin_session.expire_all()
    experiment = xngin_session.get_one(tables.Experiment, setup.experiment_id)
    assert experiment.state == ExperimentState.COMMITTED
    assert experiment.datasource_table == tab_ids[0]
    assert len(set(xngin_session.scalars(sa.select(tables.Event.id))) - events_before) == 1
