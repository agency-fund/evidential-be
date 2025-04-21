import base64
import datetime
import json
import uuid
from functools import partial

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session
from xngin.apiserver import conftest
from xngin.apiserver import main as main_module
from xngin.apiserver.api_types import DataType, ExperimentAnalysis
from xngin.apiserver.dependencies import xngin_db_session
from xngin.apiserver.main import app
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.models.tables import (
    ArmAssignment,
    ArmTable,
    Experiment,
    Organization,
    User,
)
from xngin.apiserver.routers import oidc_dependencies
from xngin.apiserver.routers.admin_api_types import (
    CreateDatasourceRequest,
    CreateDatasourceResponse,
    CreateParticipantsTypeRequest,
    CreateParticipantsTypeResponse,
    FieldMetadata,
    InspectDatasourceResponse,
    InspectDatasourceTableResponse,
    ListDatasourcesResponse,
    ListParticipantsTypeResponse,
    UpdateDatasourceRequest,
    UpdateParticipantsTypeRequest,
    UpdateParticipantsTypeResponse,
)
from xngin.apiserver.routers.experiments_api_types import (
    CreateExperimentRequest,
    CreateExperimentWithAssignmentResponse,
    ExperimentConfig,
    GetExperimentAssignmentsResponse,
    ListExperimentsResponse,
)
from xngin.apiserver.routers.oidc_dependencies import (
    PRIVILEGED_EMAIL,
    PRIVILEGED_TOKEN_FOR_TESTING,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.settings import (
    BqDsn,
    Dsn,
    GcpServiceAccountInfo,
    ParticipantsDef,
    SheetParticipantsRef,
)
from xngin.cli.main import create_testing_dwh
from xngin.schema.schema_types import FieldDescriptor, ParticipantsSchema

SAMPLE_GCLOUD_SERVICE_ACCOUNT_KEY = {
    "auth_provider_x509_cert_url": "",
    "auth_uri": "",
    "client_email": "",
    "client_id": "",
    "client_x509_cert_url": "",
    "private_key": "",
    "private_key_id": "",
    "project_id": "",
    "token_uri": "",
    "type": "service_account",
    "universe_domain": "googleapis.com",
}

conftest.setup(app)
client = TestClient(app)

pget = partial(
    client.get, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}
)
ppost = partial(
    client.post, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}
)
ppatch = partial(
    client.patch, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}
)
pdelete = partial(
    client.delete, headers={"Authorization": f"Bearer {PRIVILEGED_TOKEN_FOR_TESTING}"}
)
uget = partial(
    client.get, headers={"Authorization": f"Bearer {UNPRIVILEGED_TOKEN_FOR_TESTING}"}
)


@pytest.fixture(name="db_session", scope="module")
def fixture_db_session():
    session = next(app.dependency_overrides[xngin_db_session]())
    yield session


@pytest.fixture(autouse=True, scope="function")
def fixture_teardown(db_session: Session):
    # setup here
    yield
    # teardown here
    # Rollback any pending transactions that may have been hanging due to an exception.
    db_session.rollback()
    # Clean up objects created in each test by truncating tables and leveraging cascade.
    db_session.query(Organization).delete()
    db_session.query(User).delete()
    db_session.commit()
    db_session.close()


@pytest.fixture(scope="module", autouse=True)
def enable_apis_under_test():
    # TODO: Calling enable_*() has the minor side effect of enabling these APIs for all subsequent unit tests.
    main_module.enable_oidc_api()
    main_module.enable_admin_api()
    oidc_dependencies.TESTING_TOKENS_ENABLED = True


@pytest.fixture(name="testing_datasource_with_inline_schema")
def fixture_testing_datasource_with_inline_schema(db_session):
    """Create a fake remote datasource using an inline schema for the participants config."""
    # First create a datasource to maintain proper referential integrity, but with a local config
    # so we know we can read our dwh data. Also populate with an inline schema to test admin.
    ds_with_inlined_shema = conftest.get_settings_datasource(
        "testing-inline-schema"
    ).config
    return conftest.make_datasource_metadata(
        db_session,
        datasource_id_for_config="testing",
        participants_def_list=[ds_with_inlined_shema.participants[0]],
    )


@pytest.fixture(name="testing_datasource_with_user_added")
def fixture_testing_datasource_with_user_added(testing_datasource_with_inline_schema):
    """Add the privileged user to the test ds's organization so we can access the ds."""
    response = ppost(
        f"/v1/m/organizations/{testing_datasource_with_inline_schema.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content
    return testing_datasource_with_inline_schema


@pytest.fixture(name="testing_sheet_datasource_with_user_added")
def fixture_testing_sheet_datasource_with_user_added(testing_datasource):
    """
    Add the privileged user to the test ds's organization so we can access the ds.
    This uses a sheet participant type; most tests should use testing_datasource_with_user_added.
    """
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content
    return testing_datasource


def make_createexperimentrequest_json(participant_type: str = "test_participant_type"):
    """Create an experiment on the test ds."""
    return {
        "design_spec": {
            "experiment_name": "test",
            "description": "test",
            "experiment_type": "preassigned",
            # Attach UTC tz, but use dates_equal() to compare to respect db storage support
            "start_date": "2024-01-01T00:00:00+00:00",
            "end_date": "2024-01-02T00:00:00+00:00",
            "arms": [
                {"arm_name": "control", "arm_description": "control"},
                {"arm_name": "treatment", "arm_description": "treatment"},
            ],
            "strata_field_names": ["gender"],
            "metrics": [
                {
                    "field_name": "current_income",
                    "metric_pct_change": 0.1,
                }
            ],
        },
        "audience_spec": {
            "participant_type": participant_type,
            "filters": [],
        },
    }


def make_insertable_experiment(state: ExperimentState, datasource_id="testing"):
    request = make_createexperimentrequest_json()
    return Experiment(
        id=str(uuid.uuid4()),
        datasource_id=datasource_id,
        state=state,
        start_date=datetime.datetime.fromisoformat(
            request["design_spec"]["start_date"]
        ),
        end_date=datetime.datetime.fromisoformat(request["design_spec"]["end_date"]),
        design_spec=request["design_spec"],
        audience_spec=request["audience_spec"],
        power_analyses=None,
        assign_summary=None,
    )


def make_arms_from_experiment(experiment: Experiment, organization_id: str):
    return [
        ArmTable(
            id=arm["arm_id"],
            experiment_id=experiment.id,
            name=arm["arm_name"],
            description=arm["arm_description"],
            organization_id=organization_id,
        )
        for arm in experiment.design_spec["arms"]
    ]


@pytest.fixture(name="testing_experiment")
def fixture_testing_experiment(db_session, testing_datasource_with_user_added):
    """Create an experiment on a test inline schema datasource with proper user permissions."""
    datasource = testing_datasource_with_user_added.ds

    experiment = make_insertable_experiment(
        ExperimentState.COMMITTED,
        datasource_id=datasource.id,
    )
    # Fake arm_ids in the design_spec since we're not using the admin API to create the experiment.
    for arm in experiment.design_spec["arms"]:
        if "arm_id" not in arm:
            arm["arm_id"] = str(uuid.uuid4())
    db_session.add(experiment)
    # Create ArmTable instances for each arm in the experiment
    db_arms = make_arms_from_experiment(experiment, datasource.organization_id)
    db_session.add_all(db_arms)
    # Add fake assignments for each arm for real participant ids in our test data.
    arm_ids = [arm["arm_id"] for arm in experiment.design_spec["arms"]]
    for i in range(10):
        assignment = ArmAssignment(
            experiment_id=str(experiment.id),
            participant_id=str(i),
            participant_type=experiment.get_audience_spec().participant_type,
            arm_id=arm_ids[i % 2],  # Alternate between the two arms
            strata={},
        )
        db_session.add(assignment)
    db_session.commit()
    return experiment


def test_list_orgs_unauthenticated():
    response = client.get("/v1/m/organizations")
    assert response.status_code == 403, response.content


def test_list_orgs_privileged(testing_datasource):
    response = pget("/v1/m/organizations")
    assert response.status_code == 200, response.content


def test_list_orgs_unprivileged(testing_datasource):
    response = uget("/v1/m/organizations")
    assert response.status_code == 403, response.content


def test_lifecycle(testing_datasource):
    """Exercises the admin API methods that can operate purely in-process w/o an external database."""
    # Add privileged user to existing organization
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content

    # Add unprivileged user to existing organization
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": UNPRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content

    # List organizations
    response = pget(
        "/v1/m/organizations",
    )
    assert response.status_code == 200, response.content
    # user has their own org created upon account creation, and we created another for it.
    assert len(response.json()["items"]) == 2
    assert testing_datasource.org.id in {o["id"] for o in response.json()["items"]}

    # Create datasource
    response = ppost(
        "/v1/m/datasources",
        content=CreateDatasourceRequest(
            organization_id=testing_datasource.org.id,
            name="test remote ds",
            # These settings correspond to the Postgres spun up in GHA or via localpg.py.
            dwh=Dsn(
                driver="postgresql+psycopg",
                host="127.0.0.1",
                user="postgres",
                port=5499,
                password=SecretStr("postgres"),
                dbname="postgres",
                sslmode="disable",
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    parsed = CreateDatasourceResponse.model_validate(response.json())
    datasource_id = parsed.id

    # List datasources
    response = pget(f"/v1/m/organizations/{testing_datasource.org.id}/datasources")
    assert response.status_code == 200, response.content
    parsed = ListDatasourcesResponse.model_validate(response.json())
    assert len(parsed.items) == 2
    assert {i.driver for i in parsed.items} == {
        "postgresql+psycopg2",
        "postgresql+psycopg",
    }
    assert parsed.items[0].organization_id == parsed.items[1].organization_id

    # Update datasource name
    response = ppatch(
        f"/v1/m/datasources/{datasource_id}",
        content=UpdateDatasourceRequest(
            name="updated name",
        ).model_dump_json(),
    )
    assert response.status_code == 204, response.content

    # List datasources to confirm update
    response = pget(f"/v1/m/organizations/{testing_datasource.org.id}/datasources")
    assert response.status_code == 200, response.content
    assert "updated name" in {i["name"] for i in response.json()["items"]}, (
        response.json()
    )

    # Update DWH on the datasource
    response = ppatch(
        f"/v1/m/datasources/{datasource_id}",
        content=UpdateDatasourceRequest(
            dwh=BqDsn(
                driver="bigquery",
                project_id="123456",
                dataset_id="ds",
                credentials=GcpServiceAccountInfo(
                    type="serviceaccountinfo",
                    content_base64=base64.b64encode(
                        json.dumps(SAMPLE_GCLOUD_SERVICE_ACCOUNT_KEY).encode("utf-8")
                    ).decode(),
                ),
            )
        ).model_dump_json(),
    )
    assert response.status_code == 204, response.content

    # List datasources to confirm update
    response = pget(
        f"/v1/m/organizations/{testing_datasource.org.id}/datasources",
    )
    assert response.status_code == 200, response.content
    assert "bigquery" in {i["driver"] for i in response.json()["items"]}, (
        response.json()
    )

    # Get participants (sheet version)
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/test_participant_type",
    )
    assert response.status_code == 200, response.content
    parsed = SheetParticipantsRef.model_validate(response.json())
    assert parsed.participant_type == "test_participant_type"
    assert parsed.sheet.worksheet == "Sheet1"

    # Create participant
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants",
        content=CreateParticipantsTypeRequest(
            participant_type="newpt",
            schema_def=ParticipantsSchema(
                table_name="newps",
                fields=[
                    FieldDescriptor(
                        field_name="newf",
                        data_type=DataType.INTEGER,
                        description="test",
                        is_unique_id=True,
                        is_strata=False,
                        is_filter=False,
                        is_metric=False,
                    )
                ],
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    parsed = CreateParticipantsTypeResponse.model_validate(response.json())
    assert parsed.participant_type == "newpt"

    # List participants
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants",
    )
    assert response.status_code == 200, response.content
    parsed = ListParticipantsTypeResponse.model_validate(response.json())
    assert len(parsed.items) == 2, parsed

    # Update participant
    response = ppatch(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/newpt",
        content=UpdateParticipantsTypeRequest(
            participant_type="renamedpt"
        ).model_dump_json(),
    )
    assert response.status_code == 200
    parsed = UpdateParticipantsTypeResponse.model_validate(response.json())
    assert parsed.participant_type == "renamedpt"

    # List participants (again)
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/participants")
    assert response.status_code == 200, response.content
    parsed = ListParticipantsTypeResponse.model_validate(response.json())
    assert len(parsed.items) == 2, parsed

    # Get the named participant type
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/renamedpt",
    )
    assert response.status_code == 200, response.content
    parsed = ParticipantsDef.model_validate(response.json())
    assert parsed.participant_type == "renamedpt"

    # Delete the renamed participant type.
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/renamedpt"
    )
    assert response.status_code == 204, response.content

    # Get the named participant type after it has been deleted
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/renamedpt"
    )
    assert response.status_code == 404, response.content

    # Delete the testing participant type.
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/test_participant_type",
    )
    assert response.status_code == 204, response.content

    # Delete the testing participant type a 2nd time.
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/test_participant_type",
    )
    assert response.status_code == 404, response.content

    # Delete datasources
    response = pdelete(f"/v1/m/datasources/{testing_datasource.ds.id}")
    assert response.status_code == 204, response.content


def test_create_participants_type_invalid(testing_datasource):
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants",
        content=CreateParticipantsTypeRequest.model_construct(
            participant_type="newpt",
            schema_def=ParticipantsSchema.model_construct(
                table_name="newps",
                fields=[
                    FieldDescriptor(
                        field_name="newf",
                        data_type=DataType.INTEGER,
                        description="test",
                        is_unique_id=False,
                        is_strata=False,
                        is_filter=False,
                        is_metric=False,
                    )
                ],
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 422, response.content
    assert "no columns marked as unique ID." in response.json()["detail"][0]["msg"], (
        response.content
    )


def test_lifecycle_with_pg(testing_datasource):
    """Exercises the admin API methods that require an external database."""
    # Add the privileged user to the organization.
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content

    # Populate the testing data warehouse.
    create_testing_dwh(dsn=testing_datasource.dsn, nrows=100)

    # Inspect the datasource.
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/inspect")
    assert response.status_code == 200, response.content
    datasource_inspection = InspectDatasourceResponse.model_validate(response.json())
    assert datasource_inspection.tables == ["dwh"], response.json()

    # Inspect one table in the datasource.
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/inspect/dwh")
    assert response.status_code == 200, response.content
    table_inspection = InspectDatasourceTableResponse.model_validate(response.json())
    assert table_inspection == InspectDatasourceTableResponse(
        # Note: create_inspect_table_response_from_table() doesn't explicitly check for uniqueness.
        detected_unique_id_fields=["id", "uuid_filter"],
        fields=[
            FieldMetadata(
                field_name="baseline_income", data_type=DataType.NUMERIC, description=""
            ),
            FieldMetadata(
                field_name="current_income", data_type=DataType.NUMERIC, description=""
            ),
            FieldMetadata(
                field_name="ethnicity",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(
                field_name="first_name",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(
                field_name="gender",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(field_name="id", data_type=DataType.BIGINT, description=""),
            FieldMetadata(
                field_name="income", data_type=DataType.NUMERIC, description=""
            ),
            FieldMetadata(
                field_name="is_engaged", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="is_onboarded", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="is_recruited", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="is_registered", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="is_retained", data_type=DataType.BOOLEAN, description=""
            ),
            FieldMetadata(
                field_name="last_name",
                data_type=DataType.CHARACTER_VARYING,
                description="",
            ),
            FieldMetadata(
                field_name="potential_0", data_type=DataType.NUMERIC, description=""
            ),
            FieldMetadata(
                field_name="potential_1", data_type=DataType.BIGINT, description=""
            ),
            FieldMetadata(
                field_name="sample_date", data_type=DataType.DATE, description=""
            ),
            FieldMetadata(
                field_name="sample_timestamp",
                data_type=DataType.TIMESTAMP_WITHOUT_TIMEZONE,
                description="",
            ),
            # TODO: https://github.com/agency-fund/xngin/issues/337
            FieldMetadata(
                field_name="timestamp_with_tz",
                data_type=DataType.TIMESTAMP_WITHOUT_TIMEZONE,
                description="",
            ),
            FieldMetadata(
                field_name="uuid_filter", data_type=DataType.UUID, description=""
            ),
        ],
    )

    # Create participant
    participant_type = "participant_type_dwh"
    response = ppost(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants",
        content=CreateParticipantsTypeRequest(
            participant_type=participant_type,
            schema_def=ParticipantsSchema(
                table_name="dwh",
                fields=[
                    FieldDescriptor(
                        field_name="id",
                        data_type=DataType.INTEGER,
                        description="test",
                        is_unique_id=True,
                        is_strata=False,
                        is_filter=False,
                        is_metric=False,
                    ),
                    FieldDescriptor(
                        field_name="current_income",
                        data_type=DataType.NUMERIC,
                        description="test",
                        is_unique_id=False,
                        is_strata=False,
                        is_filter=False,
                        is_metric=True,
                    ),
                ],
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    created_participant_type = CreateParticipantsTypeResponse.model_validate(
        response.json()
    )
    assert created_participant_type.participant_type == participant_type

    # Create experiment using that participant type.
    response = ppost(
        f"/v1/m/experiments/{testing_datasource.ds.id}/with-assignment",
        params={"chosen_n": 100},
        json=make_createexperimentrequest_json(participant_type),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentWithAssignmentResponse.model_validate(
        response.json()
    )
    parsed_experiment_id = created_experiment.design_spec.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Get that experiment.
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}"
    )
    assert response.status_code == 200, response.content
    experiment_config = ExperimentConfig.model_validate(response.json())
    assert experiment_config.design_spec.experiment_id == parsed_experiment_id

    # List experiments.
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/experiments")
    assert response.status_code == 200, response.content
    experiment_list = ListExperimentsResponse.model_validate(response.json())
    assert len(experiment_list.items) == 1, experiment_list
    experiment_config = experiment_list.items[0]
    assert experiment_config.design_spec.experiment_id == parsed_experiment_id

    # Analyze experiment
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}/analyze"
    )
    assert response.status_code == 200, response.content
    experiment_analysis = ExperimentAnalysis.model_validate(response.json())
    assert experiment_analysis.experiment_id == parsed_experiment_id

    # Get assignments for the experiment.
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}/assignments"
    )
    assert response.status_code == 200, response.content
    assignments = GetExperimentAssignmentsResponse.model_validate(response.json())
    assert assignments.experiment_id == parsed_experiment_id
    assert assignments.sample_size == 100
    assert assignments.balance_check is not None
    assert len(assignments.assignments) == 100
    assert {arm.arm_name for arm in assignments.assignments} == {"control", "treatment"}
    assert {arm.arm_id for arm in assignments.assignments} == parsed_arm_ids

    # Delete the experiment.
    response = pdelete(
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments/{parsed_experiment_id}"
    )
    assert response.status_code == 204, response.content


def test_create_experiment_with_assignment_validation_errors(
    testing_datasource_with_user_added,
    testing_sheet_datasource_with_user_added,
):
    """Test LateValidationError cases in create_experiment_with_assignment."""
    testing_datasource = testing_datasource_with_user_added

    # Create a basic experiment request
    # Test 1: UUIDs present in design spec trigger LateValidationError
    base_request = make_createexperimentrequest_json("test_participant_type")
    base_request["design_spec"]["experiment_id"] = (
        "123e4567-e89b-12d3-a456-426614174000"
    )
    response = ppost(
        f"/v1/m/experiments/{testing_datasource.ds.id}/with-assignment",
        params={"chosen_n": 100},
        json=base_request,
    )
    assert response.status_code == 422, response.content
    assert "UUIDs must not be set" in response.json()["message"]

    # Test 2: Invalid participants config (sheet instead of schema)
    # This datasource is loaded with a "remote" config from xngin.testing.settings.json, but
    # the associated participants config is of type "sheet".
    response = ppost(
        f"/v1/m/experiments/{testing_sheet_datasource_with_user_added.ds.id}/with-assignment",
        params={"chosen_n": 100},
        json=make_createexperimentrequest_json(),
    )
    assert response.status_code == 422, response.content
    assert "Participants must be of type schema" in response.json()["message"]


def test_create_experiment_with_assignment_using_inline_schema_ds(
    db_session, testing_datasource_with_user_added, use_deterministic_random
):
    datasource_id = testing_datasource_with_user_added.ds.id
    base_request_json = make_createexperimentrequest_json("test_participant_type")
    base_request = CreateExperimentRequest.model_validate(base_request_json)

    response = ppost(
        f"/v1/m/experiments/{datasource_id}/with-assignment",
        params={"chosen_n": 100, "random_state": 42},
        json=base_request_json,
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentWithAssignmentResponse.model_validate(
        response.json()
    )
    parsed_experiment_id = created_experiment.design_spec.experiment_id
    assert parsed_experiment_id is not None
    parsed_arm_ids = {arm.arm_id for arm in created_experiment.design_spec.arms}
    assert len(parsed_arm_ids) == 2

    # Verify basic response
    assert created_experiment.design_spec.experiment_id is not None
    assert created_experiment.design_spec.arms[0].arm_id is not None
    assert created_experiment.design_spec.arms[1].arm_id is not None
    assert created_experiment.state == ExperimentState.ASSIGNED
    assign_summary = created_experiment.assign_summary
    assert assign_summary.sample_size == 100
    assert assign_summary.balance_check.balance_ok is True

    # Check if the representations are equivalent
    # scrub the uuids from the config for comparison
    actual_design_spec = created_experiment.design_spec.model_copy(deep=True)
    actual_design_spec.experiment_id = None
    actual_design_spec.arms[0].arm_id = None
    actual_design_spec.arms[1].arm_id = None
    assert actual_design_spec == base_request.design_spec
    assert created_experiment.audience_spec == base_request.audience_spec
    assert created_experiment.power_analyses == base_request.power_analyses

    experiment_id = created_experiment.design_spec.experiment_id
    (arm1_id, arm2_id) = [
        str(arm.arm_id) for arm in created_experiment.design_spec.arms
    ]

    # Verify database state using the ids in the returned DesignSpec.
    experiment = db_session.scalars(
        select(Experiment).where(Experiment.id == str(experiment_id))
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == datasource_id
    assert conftest.dates_equal(
        experiment.start_date, base_request.design_spec.start_date
    )
    assert conftest.dates_equal(experiment.end_date, base_request.design_spec.end_date)
    # Verify assignments were created
    assignments = db_session.scalars(
        select(ArmAssignment).where(ArmAssignment.experiment_id == str(experiment_id))
    ).all()
    assert len(assignments) == 100, {
        e.name: getattr(experiment, e.name) for e in Experiment.__table__.columns
    }

    # Check one assignment to see if it looks roughly right
    sample_assignment: ArmAssignment = assignments[0]
    assert sample_assignment.participant_type == "test_participant_type"
    assert sample_assignment.experiment_id == str(experiment_id)
    assert sample_assignment.arm_id in {arm1_id, arm2_id}
    for stratum in sample_assignment.strata:
        assert stratum["field_name"] in {"current_income", "gender"}

    # Check for approximate balance in arm assignment
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 5  # Allow some wiggle room


def test_experiments_analyze(testing_experiment):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze"
    )

    assert response.status_code == 200, response.content
    experiment_analysis = ExperimentAnalysis.model_validate(response.json())
    # ExperimentAnalysis uses real native uuids, whereas the returned database model are actually
    # treated as strings by sqlalchemy given our table definition, so compare as strings.
    assert str(experiment_analysis.experiment_id) == str(experiment_id)
    assert len(experiment_analysis.metric_analyses) == 1
    # Verify that only the first arm is marked as baseline by default
    metric_analysis = experiment_analysis.metric_analyses[0]
    baseline_arms = [arm for arm in metric_analysis.arm_analyses if arm.is_baseline]
    assert len(baseline_arms) == 1
    assert baseline_arms[0].is_baseline
    for analysis in experiment_analysis.metric_analyses:
        # Verify arm_ids match the database model.
        assert {str(arm.arm_id) for arm in analysis.arm_analyses} == {
            str(arm.arm_id) for arm in testing_experiment.get_arms()
        }


@pytest.mark.parametrize(
    "endpoint,initial_state,expected_status,expected_detail",
    [
        ("commit", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("commit", ExperimentState.COMMITTED, 304, None),  # No-op
        ("commit", ExperimentState.DESIGNING, 403, "Invalid state: designing"),
        ("commit", ExperimentState.ABORTED, 403, "Invalid state: aborted"),
        ("abandon", ExperimentState.DESIGNING, 204, None),  # Success case
        ("abandon", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("abandon", ExperimentState.ABANDONED, 304, None),  # No-op
        ("abandon", ExperimentState.COMMITTED, 403, "Invalid state: committed"),
    ],
)
def test_admin_experiment_state_setting(
    db_session,
    testing_datasource_with_user_added,
    endpoint,
    initial_state,
    expected_status,
    expected_detail,
):
    # Initialize our state with an existing experiment who's state we want to modify.
    datasource_id = testing_datasource_with_user_added.ds.id
    experiment = make_insertable_experiment(initial_state, datasource_id=datasource_id)
    db_session.add(experiment)
    db_session.commit()

    response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment.id!s}/{endpoint}"
    )

    # Verify
    assert response.status_code == expected_status
    # If success case, verify state was updated
    if expected_status == 204:
        expected_state = (
            ExperimentState.ABANDONED
            if endpoint == "abandon"
            else ExperimentState.COMMITTED
        )
        db_session.refresh(experiment)
        assert experiment.state == expected_state
    # If failure case, verify the error message
    if expected_detail:
        assert response.json()["detail"] == expected_detail
