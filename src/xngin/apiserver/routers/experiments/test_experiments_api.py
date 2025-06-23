import glob
import json
import re
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from json import JSONDecodeError
from pathlib import Path

import pytest
from deepdiff import DeepDiff
from loguru import logger
from sqlalchemy import delete, select

from xngin.apiserver import conftest, constants, flags
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.models import tables
from xngin.apiserver.models.enums import ExperimentState, StopAssignmentReason
from xngin.apiserver.models.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.routers.common_api_types import (
    CreateExperimentResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
    PreassignedExperimentSpec,
)
from xngin.apiserver.routers.experiments.test_experiments_common import (
    insert_experiment_and_arms,
    make_create_preassigned_experiment_request,
)
from xngin.apiserver.routers.stateless.stateless_api import (
    CommonQueryParams,
    get_participants_config_and_schema,
)
from xngin.apiserver.settings import ParticipantsDef
from xngin.apiserver.testing.assertions import assert_same
from xngin.apiserver.testing.xurl import Xurl


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
async def fixture_teardown(xngin_session):
    try:
        # setup here
        yield
    finally:
        # teardown here
        # Rollback any pending transactions that may have been hanging due to an exception.
        await xngin_session.rollback()
        # Ensure we're not using stale cache settings (possible if not using an ephemeral app db).
        await xngin_session.execute(delete(tables.CacheTable))
        await xngin_session.commit()


async def test_datasource_dependency_falls_back_to_xngin_db(
    xngin_session, testing_datasource
):
    local_cache = GSheetCache(xngin_session)

    participants_cfg_sheet, schema_sheet = await get_participants_config_and_schema(
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
    await xngin_session.commit()
    # ...and verify we retrieve that correctly.
    participants_cfg, schema = await get_participants_config_and_schema(
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
def test_api(
    script, datasource_id, update_api_tests_flag, use_deterministic_random, client_v1
):
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
    response = client_v1.request(
        xurl.method, xurl.url, headers=headers, content=xurl.body
    )
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


def test_create_experiment_impl_invalid_design_spec(client_v1):
    """Test creating an experiment and saving assignments to the database."""
    request = make_create_preassigned_experiment_request(with_ids=True)

    response = client_v1.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100},
        headers={constants.HEADER_CONFIG_ID: "testing"},
        content=request.model_dump_json(),
    )
    assert response.status_code == 422, request
    assert "UUIDs must not be set" in response.json()["message"]


async def test_create_experiment_with_assignment_sl(
    xngin_session, use_deterministic_random, client_v1
):
    """Test creating an experiment and saving assignments to the database."""
    # First create a datasource to maintain proper referential integrity, but with a local config so we know we can read our dwh data.
    ds_metadata = await conftest.make_datasource_metadata(
        xngin_session, datasource_id="testing"
    )
    request = make_create_preassigned_experiment_request()

    response = client_v1.post(
        "/experiments/with-assignment",
        params={"chosen_n": 100},
        headers={
            constants.HEADER_CONFIG_ID: ds_metadata.ds.id,
            constants.HEADER_API_KEY: ds_metadata.key,
        },
        content=request.model_dump_json(),
    )

    # Verify basic response
    assert response.status_code == 200, request
    experiment_config = CreateExperimentResponse.model_validate(response.json())
    assert experiment_config.design_spec.experiment_id is not None
    assert experiment_config.design_spec.arms[0].arm_id is not None
    assert experiment_config.design_spec.arms[1].arm_id is not None
    assert experiment_config.datasource_id == ds_metadata.ds.id
    assert experiment_config.state == ExperimentState.ASSIGNED


async def test_list_experiments_sl_without_api_key(
    xngin_session, testing_datasource, client_v1
):
    """Tests that listing experiments tied to a db datasource requires an API key."""
    await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED
    )

    response = client_v1.get(
        "/experiments",
        headers={constants.HEADER_CONFIG_ID: testing_datasource.ds.id},
    )
    assert response.status_code == 403
    assert response.json()["message"] == "API key missing or invalid."


async def test_list_experiments_sl_with_api_key(
    xngin_session, testing_datasource, client_v1
):
    """Tests that listing experiments tied to a db datasource with an API key works."""
    expected_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.ASSIGNED
    )

    response = client_v1.get(
        "/experiments",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response.status_code == 200
    experiments = ListExperimentsResponse.model_validate(response.json())
    assert len(experiments.items) == 1
    assert experiments.items[0].state == ExperimentState.ASSIGNED
    expected_design_spec = ExperimentStorageConverter(
        expected_experiment
    ).get_design_spec()
    diff = DeepDiff(expected_design_spec, experiments.items[0].design_spec)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


async def test_get_experiment(xngin_session, testing_datasource, client_v1):
    new_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        state=ExperimentState.DESIGNING,
    )

    response = client_v1.get(
        f"/experiments/{new_experiment.id!s}",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )

    assert response.status_code == 200, response.content

    experiment_json = response.json()
    assert experiment_json["datasource_id"] == new_experiment.datasource_id
    assert experiment_json["state"] == new_experiment.state
    actual = PreassignedExperimentSpec.model_validate(experiment_json["design_spec"])
    expected = ExperimentStorageConverter(new_experiment).get_design_spec()
    diff = DeepDiff(actual, expected)
    assert not diff, f"Objects differ:\n{diff.pretty()}"


def test_get_experiment_assignments_not_found(testing_datasource, client_v1):
    """Test getting assignments for a non-existent experiment."""
    response = client_v1.get(
        f"/experiments/{tables.experiment_id_factory()}/assignments",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response.status_code == 404, response.json()
    assert response.json()["detail"] == "Experiment not found or not authorized."


async def test_get_experiment_assignments_wrong_datasource(
    xngin_session, testing_datasource, client_v1
):
    """Test getting assignments for an experiment from a different datasource."""
    # Create experiment in one datasource
    experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, state=ExperimentState.COMMITTED
    )
    # Make a *different* datasource and API key to query with
    metadata = await conftest.make_datasource_metadata(xngin_session, name="wrong ds")

    # Try to get testing_datasource's experiment from another datasource's key.
    response = client_v1.get(
        f"/experiments/{experiment.id!s}/assignments",
        headers={constants.HEADER_API_KEY: metadata.key},
    )
    assert response.status_code == 404, response.json()
    assert response.json()["detail"] == "Experiment not found or not authorized."


async def test_get_assignment_for_participant_with_apikey_preassigned(
    xngin_session, testing_datasource, client_v1
):
    preassigned_experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds
    )
    assignment = tables.ArmAssignment(
        experiment_id=preassigned_experiment.id,
        participant_id="assigned_id",
        participant_type=preassigned_experiment.participant_type,
        arm_id=preassigned_experiment.arms[0].id,
        strata=[],
    )
    xngin_session.add(assignment)
    await xngin_session.commit()

    response = client_v1.get(
        f"/experiments/{preassigned_experiment.id!s}/assignments/unassigned_id?random_state=42",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == preassigned_experiment.id
    assert parsed.participant_id == "unassigned_id"
    assert parsed.assignment is None

    response = client_v1.get(
        f"/experiments/{preassigned_experiment.id!s}/assignments/assigned_id",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == preassigned_experiment.id
    assert parsed.participant_id == "assigned_id"
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == "control"


async def test_get_assignment_for_participant_with_apikey_online(
    xngin_session, testing_datasource, client_v1
):
    """Test endpoint that gets an assignment for a participant via API key."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type="online",
    )

    response = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    arms_map = {arm.id: arm.name for arm in online_experiment.arms}
    assert parsed.assignment is not None
    assert parsed.assignment.arm_name == arms_map[str(parsed.assignment.arm_id)]
    assert parsed.assignment.arm_name == "control"
    assert not parsed.assignment.strata

    # Test that we get the same assignment for the same participant.
    response2 = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
    )
    assert response2.status_code == 200
    assert response2.json() == response.json()

    # Make sure there's only one db entry.
    assignment = (
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(
                tables.ArmAssignment.experiment_id == online_experiment.id
            )
        )
    ).one()
    assert assignment.participant_id == "1"
    assert assignment.arm_id == str(parsed.assignment.arm_id)

    # Verify no update to experiment lifecycle info.
    assert assignment.experiment.stopped_assignments_at is None
    assert assignment.experiment.stopped_assignments_reason is None


async def test_get_assignment_for_participant_with_apikey_online_dont_create(
    xngin_session, testing_datasource, client_v1
):
    """Verify endpoint doesn't create an assignment when create_if_none=False."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type="online",
    )

    response = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={constants.HEADER_API_KEY: testing_datasource.key},
        params={"create_if_none": False},
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None


async def test_get_assignment_for_participant_with_apikey_online_past_end_date(
    xngin_session, testing_datasource, client_v1
):
    """Verify endpoint doesn't create an assignment for an online experiment that has ended."""
    online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type="online",
        end_date=datetime.now(UTC) - timedelta(days=1),
    )

    response = client_v1.get(
        f"/experiments/{online_experiment.id!s}/assignments/1",
        headers={
            constants.HEADER_CONFIG_ID: testing_datasource.ds.id,
            constants.HEADER_API_KEY: testing_datasource.key,
        },
    )
    assert response.status_code == 200
    parsed = GetParticipantAssignmentResponse.model_validate_json(response.text)
    assert parsed.experiment_id == online_experiment.id
    assert parsed.participant_id == "1"
    assert parsed.assignment is None

    # Verify side effect to experiment lifecycle info.
    await xngin_session.refresh(online_experiment)
    assert online_experiment.stopped_assignments_at is not None
    assert online_experiment.stopped_assignments_reason == StopAssignmentReason.END_DATE
