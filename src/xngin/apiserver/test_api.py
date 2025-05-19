import glob
import json
import re
import shutil
import tempfile
from json import JSONDecodeError
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from xngin.apiserver import conftest, constants, flags
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.main import app
from xngin.apiserver.models.tables import CacheTable
from xngin.apiserver.routers.stateless_api import (
    CommonQueryParams,
    get_participants_config_and_schema,
)
from xngin.apiserver.settings import ParticipantsDef
from xngin.apiserver.testing.assertions import assert_same
from xngin.apiserver.testing.xurl import Xurl

conftest.setup(app)
client = TestClient(app)
client.base_url = str(client.base_url) + constants.API_PREFIX_V1


def mark_nondeterministic_tests(c):
    """Marks known nondeterministic tests with a mark that allows us to skip them except when requested."""
    return c


API_TESTS = [
    mark_nondeterministic_tests(c)
    for c in glob.glob(str(Path(__file__).parent / "testdata/*.xurl"))
]


def trunc(s, n=4096):
    """Truncates a string at a length that is usable in unit tests."""
    if isinstance(s, bytes):
        s = str(s)
    if len(s) > n:
        s = s[:n] + "..."
    return s


@pytest.fixture(name="update_api_tests_flag")
def fixture_update_api_tests_flag(pytestconfig):
    """Returns true iff the UPDATE_API_TESTS environment variable resembles a truthy value."""
    return flags.UPDATE_API_TESTS


@pytest.fixture(autouse=True)
def fixture_teardown(xngin_session):
    # setup here
    yield
    # teardown here
    # Rollback any pending transactions that may have been hanging due to an exception.
    xngin_session.rollback()
    # Ensure we're not using stale cache settings (possible if not using an ephemeral app db).
    xngin_session.query(CacheTable).delete()
    xngin_session.commit()


def test_datasource_dependency_falls_back_to_xngin_db(
    xngin_session, testing_datasource
):
    local_cache = GSheetCache(xngin_session)

    participants_cfg_sheet, schema_sheet = get_participants_config_and_schema(
        commons=CommonQueryParams("test_participant_type"),
        datasource_config=conftest.get_settings_for_test()
        .get_datasource("testing-remote")
        .config,
        gsheets=local_cache,
    )
    assert participants_cfg_sheet.type == "sheet"

    # Now store a version that inlines the schema...
    participants_def = ParticipantsDef(
        type="schema",
        participant_type=participants_cfg_sheet.participant_type,
        table_name=schema_sheet.table_name,
        fields=schema_sheet.fields,
    )
    config = testing_datasource.ds.get_config()
    config.participants = [participants_def]
    testing_datasource.ds.set_config(config)
    xngin_session.commit()
    # ...and verify we retrieve that correctly.
    participants_cfg, schema = get_participants_config_and_schema(
        commons=CommonQueryParams("test_participant_type"),
        datasource_config=testing_datasource.ds.get_config(),
        gsheets=local_cache,
    )
    assert participants_cfg.type == "schema"
    assert_same(
        schema_sheet.model_dump(),
        schema.model_dump(),
        deepdiff_kwargs={"exclude_paths": ["participant_type", "type"]},
    )


API_TESTS_X_DATASOURCE = zip(
    API_TESTS * 2,
    [None] * len(API_TESTS) + ["testing-inline-schema"] * len(API_TESTS),
    strict=False,
)


@pytest.mark.parametrize("script,datasource_id", API_TESTS_X_DATASOURCE)
def test_api(script, datasource_id, update_api_tests_flag, use_deterministic_random):
    """Runs all the API_TESTS test scripts using the datasource specified in param or file if None.

    Test scripts may omit asserting equality of actual response and expected response on specific paths. For example:

    [test_api]
    {
        "deepdiff_kwargs": {
            "exclude_paths": [
                "root['assignments']",
                "root['f_statistic']",
                "root['p_value']"
            ]
        }
    }

    Furthermore, arbitrary args can be passed to deepdiff.diff via the deepdiff_kwargs key.
    """

    def parse_trailer(trailer):
        """Extracts the value of deepdiff_kwargs from the JSON following any [test_api] header, or empty dict."""
        if not trailer:
            return {}
        matches = re.search(r"(?s)\[test_api\]\n(?P<args>.*)", trailer)
        if not matches:
            return {}
        try:
            config = json.loads(matches.group("args"))
            return config["deepdiff_kwargs"]
        except JSONDecodeError as e:
            pytest.fail(
                f"The script {script} contains an invalid JSON body in the [test_api] trailer: {e!s}"
            )

    with open(script, encoding="utf-8") as f:
        contents = f.read()
    xurl = Xurl.from_script(contents)
    headers = xurl.headers
    # Override datasource-id header to also test inline schemas; should have the same responses as
    # using the old settings.json configuration approach.
    if datasource_id is not None:
        assert constants.HEADER_CONFIG_ID in headers
        headers[constants.HEADER_CONFIG_ID] = datasource_id
    response = client.request(xurl.method, xurl.url, headers=headers, content=xurl.body)
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=".xurl")  # noqa: SIM115
    # Write the actual response to a temporary file. If an exception is thrown, we optionally
    # replace the script we just executed with the new script.
    with temporary as tmpf:
        actual = xurl.model_copy()
        actual.expected_status = response.status_code
        actual.expected_response = json.dumps(response.json(), indent=2, sort_keys=True)
        tmpf.write(actual.to_script().encode("utf-8"))
    try:
        assert response.status_code == xurl.expected_status, (
            f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}"
        )
        deepdiff_kwargs = parse_trailer(xurl.trailer)
        assert_same(
            response.json(),
            json.loads(xurl.expected_response or "{}"),
            deepdiff_kwargs=deepdiff_kwargs,
            extra=f"HTTP response body: {temporary.name}\nResponse:\n{trunc(response.content)}",
        )
    except AssertionError:
        if update_api_tests_flag:
            logger.info(f"Updating API test {script}.")
            shutil.copy(temporary.name, script)
        raise
