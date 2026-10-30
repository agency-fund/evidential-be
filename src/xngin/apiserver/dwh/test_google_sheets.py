import io
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import quote

import gspread
import numpy as np
import pytest
import requests
import sqlalchemy as sa
from google.auth.credentials import AnonymousCredentials
from google.auth.exceptions import RefreshError

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
    DesignSpecMetricRequest,
    Filter,
)
from xngin.apiserver.routers.common_enums import Relation
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
        selected_gid=None,
        client=mocker.MagicMock(),
        spreadsheet=mocker.MagicMock(title="Evidential Demo Sheet"),
    )
    api.client.http_client.auth.service_account_email = "reader@evidential-tests.iam.gserviceaccount.com"

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
