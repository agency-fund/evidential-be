import base64
import datetime
import json
from functools import partial

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr, TypeAdapter
from sqlalchemy import select
from xngin.apiserver import conftest, flags
from xngin.apiserver import main as main_module
from xngin.apiserver.dns import safe_resolve
from xngin.apiserver.main import app
from xngin.apiserver.models import tables
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.models.storage_types import DesignSpecFields
from xngin.apiserver.routers import oidc_dependencies
from xngin.apiserver.routers.admin import user_from_token
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
    CreateExperimentResponse,
    ExperimentConfig,
    GetExperimentAssignmentsResponse,
    GetParticipantAssignmentResponse,
    ListExperimentsResponse,
)
from xngin.apiserver.routers.oidc_dependencies import (
    PRIVILEGED_EMAIL,
    PRIVILEGED_TOKEN_FOR_TESTING,
    TESTING_TOKENS,
    UNPRIVILEGED_EMAIL,
    UNPRIVILEGED_TOKEN_FOR_TESTING,
)
from xngin.apiserver.routers.stateless_api_types import (
    DataType,
    DesignSpec,
    ExperimentAnalysis,
    ExperimentType,
)
from xngin.apiserver.settings import (
    BqDsn,
    Dsn,
    GcpServiceAccountInfo,
    ParticipantsDef,
    SheetParticipantsRef,
    infer_table,
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
udelete = partial(
    client.delete, headers={"Authorization": f"Bearer {UNPRIVILEGED_TOKEN_FOR_TESTING}"}
)
uget = partial(
    client.get, headers={"Authorization": f"Bearer {UNPRIVILEGED_TOKEN_FOR_TESTING}"}
)


@pytest.fixture(autouse=True)
def fixture_teardown(xngin_session):
    try:
        # setup here
        yield
    finally:
        # teardown here
        # Rollback any pending transactions that may have been hanging due to an exception.
        xngin_session.rollback()
        # Clean up objects created in each test by truncating tables and leveraging cascade.
        xngin_session.query(tables.Organization).delete()
        xngin_session.query(tables.User).delete()
        xngin_session.commit()


@pytest.fixture(scope="module", autouse=True)
def enable_apis_under_test():
    # TODO: Calling enable_*() has the minor side effect of enabling these APIs for all subsequent unit tests.
    main_module.enable_oidc_api()
    main_module.enable_admin_api()
    oidc_dependencies.TESTING_TOKENS_ENABLED = True


@pytest.fixture(name="testing_datasource_with_inline_schema")
def fixture_testing_datasource_with_inline_schema(xngin_session):
    """Create a fake remote datasource using an inline schema for the participants config."""
    # First create a datasource to maintain proper referential integrity, but with a local config
    # so we know we can read our dwh data. Also populate with an inline schema to test admin.
    ds_with_inlined_shema = conftest.get_settings_datasource(
        "testing-inline-schema"
    ).config
    return conftest.make_datasource_metadata(
        xngin_session,
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


def make_createexperimentrequest_json(
    participant_type: str = "test_participant_type",
    experiment_type: str = "preassigned",
):
    """Create an experiment on the test ds."""
    return {
        "design_spec": {
            "participant_type": participant_type,
            "experiment_name": "test",
            "description": "test",
            "experiment_type": experiment_type,
            # Attach UTC tz, but use dates_equal() to compare to respect db storage support
            "start_date": "2024-01-01T00:00:00+00:00",
            "end_date": "2024-01-02T00:00:00+00:00",
            "arms": [
                {"arm_name": "control", "arm_description": "control"},
                {"arm_name": "treatment", "arm_description": "treatment"},
            ],
            "filters": [],
            "strata": [{"field_name": "gender"}],
            "metrics": [
                {
                    "field_name": "current_income",
                    "metric_pct_change": 0.1,
                }
            ],
        }
    }


def make_insertable_experiment(
    state: ExperimentState,
    datasource_id: str = "testing",
    experiment_type: ExperimentType = "preassigned",
) -> tables.Experiment:
    request = make_createexperimentrequest_json(experiment_type=experiment_type)
    design_spec: DesignSpec = TypeAdapter(DesignSpec).validate_python(
        request["design_spec"]
    )
    # TODO(qixotic): experiment_id should also be set on DesignSpec
    return tables.Experiment(
        id=tables.experiment_id_factory(),
        datasource_id=datasource_id,
        experiment_type=experiment_type,
        participant_type=design_spec.participant_type,
        name=design_spec.experiment_name,
        description=design_spec.description,
        state=state,
        start_date=datetime.datetime.fromisoformat(design_spec.start_date.isoformat()),
        end_date=datetime.datetime.fromisoformat(design_spec.end_date.isoformat()),
        power=design_spec.power,
        alpha=design_spec.alpha,
        fstat_thresh=design_spec.fstat_thresh,
        design_spec=design_spec.model_dump(mode="json"),
        design_spec_fields=DesignSpecFields(
            strata=design_spec.strata,
            metrics=design_spec.metrics,
            filters=design_spec.filters,
        ).model_dump(mode="json"),
        power_analyses=None,
    ).set_balance_check(None)


def make_arms_from_experiment(
    experiment: tables.Experiment, organization_id: str
) -> list[tables.ArmTable]:
    return [
        tables.ArmTable(
            id=arm["arm_id"],
            experiment_id=experiment.id,
            name=arm["arm_name"],
            description=arm["arm_description"],
            organization_id=organization_id,
        )
        for arm in experiment.design_spec["arms"]
    ]


def make_experiment_and_arms(
    xngin_session, datasource: tables.Datasource, experiment_type: ExperimentType
) -> tables.Experiment:
    experiment = make_insertable_experiment(
        ExperimentState.COMMITTED,
        datasource_id=datasource.id,
        experiment_type=experiment_type,
    )
    # Fake arm_ids in the design_spec since we're not using the admin API to create the experiment.
    for arm in experiment.design_spec["arms"]:
        if "arm_id" not in arm or arm["arm_id"] is None:
            arm["arm_id"] = tables.arm_id_factory()
    xngin_session.add(experiment)
    # Create ArmTable instances for each arm in the experiment
    db_arms = make_arms_from_experiment(experiment, datasource.organization_id)
    xngin_session.add_all(db_arms)
    xngin_session.commit()
    return experiment


@pytest.fixture(name="testing_experiment")
def fixture_testing_experiment(xngin_session, testing_datasource_with_user_added):
    """Create an experiment on a test inline schema datasource with proper user permissions."""
    datasource = testing_datasource_with_user_added.ds
    experiment = make_experiment_and_arms(xngin_session, datasource, "preassigned")
    # Add fake assignments for each arm for real participant ids in our test data.
    arm_ids = [arm.id for arm in experiment.arms]
    for i in range(10):
        assignment = tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_id=str(i),
            participant_type=experiment.participant_type,
            arm_id=arm_ids[i % 2],  # Alternate between the two arms
            strata=[],
        )
        xngin_session.add(assignment)
    xngin_session.commit()
    return experiment


def test_user_from_token(xngin_session):
    with pytest.raises(HTTPException, match="No user found with email") as e:
        user_from_token(xngin_session, TESTING_TOKENS[UNPRIVILEGED_TOKEN_FOR_TESTING])
    assert e.value.status_code == 403

    user = user_from_token(xngin_session, TESTING_TOKENS[PRIVILEGED_TOKEN_FOR_TESTING])
    assert user.is_privileged

    org = user.organizations[0]
    ds = org.datasources[0]
    ds_config = ds.get_config()
    pt_def = ds_config.participants[0]
    # Assert it's a "schema" type, not the old "sheets" type.
    assert isinstance(pt_def, ParticipantsDef)
    # Check auto-generated ParticipantsDef is aligned with the test dwh.
    session = ds_config.dbsession()
    sa_table = infer_table(session.get_bind(), pt_def.table_name)
    col_names = {c.name for c in sa_table.columns}
    field_names = {f.field_name for f in pt_def.fields}
    assert col_names == field_names
    for field in pt_def.fields:
        col = sa_table.columns[field.field_name]
        assert DataType.match(col.type) == field.data_type


@pytest.mark.skipif(
    flags.AIRPLANE_MODE,
    reason="This test will fail in airplane mode because airplane mode treats all Admin API calls as authenticated.",
)
def test_list_orgs_unauthenticated():
    response = client.get("/v1/m/organizations")
    assert response.status_code == 403, response.content


def test_list_orgs_privileged(testing_datasource):
    response = pget("/v1/m/organizations")
    assert response.status_code == 200, response.content


@pytest.mark.skipif(
    flags.AIRPLANE_MODE,
    reason="This test will fail in airplane mode because airplane mode treats all Admin API calls as authenticated.",
)
def test_list_orgs_unprivileged(testing_datasource):
    response = uget("/v1/m/organizations")
    assert response.status_code == 403, response.content


def test_create_datasource_invalid_dns(testing_datasource):
    """Tests that we reject insecure hostnames with a 400."""
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content

    response = ppost(
        "/v1/m/datasources",
        content=CreateDatasourceRequest(
            organization_id=testing_datasource.org.id,
            name="test remote ds",
            dwh=Dsn(
                driver="postgresql+psycopg",
                host=safe_resolve.UNSAFE_IP_FOR_TESTING,
                user="postgres",
                port=5499,
                password=SecretStr("postgres"),
                dbname="postgres",
                sslmode="disable",
            ),
        ).model_dump_json(),
    )
    assert response.status_code == 400, response.content
    assert "DNS resolution failed" in str(response.content)


def test_lifecycle(testing_datasource):
    """Exercises the admin API methods that can operate purely in-process w/o an external dwh."""
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
    datasource_response = CreateDatasourceResponse.model_validate(response.json())
    datasource_id = datasource_response.id

    # List datasources
    response = pget(f"/v1/m/organizations/{testing_datasource.org.id}/datasources")
    assert response.status_code == 200, response.content
    list_ds_response = ListDatasourcesResponse.model_validate(response.json())
    assert len(list_ds_response.items) == 2
    assert {i.driver for i in list_ds_response.items} == {
        "postgresql+psycopg2",
        "postgresql+psycopg",
    }
    assert (
        list_ds_response.items[0].organization_id
        == list_ds_response.items[1].organization_id
    )

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
    create_pt_response = CreateParticipantsTypeResponse.model_validate(response.json())
    assert create_pt_response.participant_type == "newpt"

    # List participants
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants",
    )
    assert response.status_code == 200, response.content
    list_pt_response = ListParticipantsTypeResponse.model_validate(response.json())
    assert len(list_pt_response.items) == 2, list_pt_response

    # Update participant
    response = ppatch(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/newpt",
        content=UpdateParticipantsTypeRequest(
            participant_type="renamedpt"
        ).model_dump_json(),
    )
    assert response.status_code == 200, response.content
    update_pt_response = UpdateParticipantsTypeResponse.model_validate(response.json())
    assert update_pt_response.participant_type == "renamedpt"

    # List participants (again)
    response = pget(f"/v1/m/datasources/{testing_datasource.ds.id}/participants")
    assert response.status_code == 200, response.content
    list_pt_response = ListParticipantsTypeResponse.model_validate(response.json())
    assert len(list_pt_response.items) == 2, list_pt_response

    # Get the named participant type
    response = pget(
        f"/v1/m/datasources/{testing_datasource.ds.id}/participants/renamedpt",
    )
    assert response.status_code == 200, response.content
    participants_def = ParticipantsDef.model_validate(response.json())
    assert participants_def.participant_type == "renamedpt"

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


def test_delete_datasource(testing_datasource_with_user):
    # Delete the datasource as an unprivileged user.
    response = udelete(f"/v1/m/datasources/{testing_datasource_with_user.ds.id}")
    assert response.status_code == 403, response.content

    # Delete the datasource as a privileged user.
    response = pdelete(f"/v1/m/datasources/{testing_datasource_with_user.ds.id}")
    assert response.status_code == 204, response.content

    # Delete the datasource a 2nd time.
    response = pdelete(f"/v1/m/datasources/{testing_datasource_with_user.ds.id}")
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


def test_lifecycle_with_db(testing_datasource):
    """Exercises the admin API methods that require an external database."""
    # Add the privileged user to the organization.
    response = ppost(
        f"/v1/m/organizations/{testing_datasource.org.id}/members",
        json={"email": PRIVILEGED_EMAIL},
    )
    assert response.status_code == 204, response.content

    # Populate the testing data warehouse. NOTE: This will drop and recreate the database!
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
            FieldMetadata(
                field_name="timestamp_with_tz",
                data_type=DataType.TIMESTAMP_WITH_TIMEZONE,
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
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments",
        params={"chosen_n": 100},
        json=make_createexperimentrequest_json(participant_type),
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
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
        f"/v1/m/datasources/{testing_datasource.ds.id}/experiments",
        params={"chosen_n": 100},
        json=base_request,
    )
    assert response.status_code == 422, response.content
    assert "UUIDs must not be set" in response.json()["message"]

    # Test 2: Invalid participants config (sheet instead of schema)
    # This datasource is loaded with a "remote" config from xngin.testing.settings.json, but
    # the associated participants config is of type "sheet".
    response = ppost(
        f"/v1/m/datasources/{testing_sheet_datasource_with_user_added.ds.id}/experiments",
        params={"chosen_n": 100},
        json=make_createexperimentrequest_json(),
    )
    assert response.status_code == 422, response.content
    assert "Participants must be of type schema" in response.json()["message"]


def test_create_preassigned_experiment_using_inline_schema_ds(
    xngin_session, testing_datasource_with_user_added, use_deterministic_random
):
    datasource_id = testing_datasource_with_user_added.ds.id
    base_request_json = make_createexperimentrequest_json("test_participant_type")
    base_request = CreateExperimentRequest.model_validate(base_request_json)

    response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments",
        params={"chosen_n": 100, "random_state": 42},
        json=base_request_json,
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
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
    assert assign_summary.balance_check is not None
    assert assign_summary.balance_check.balance_ok is True

    # Check if the representations are equivalent
    # scrub the ids from the config for comparison
    actual_design_spec = created_experiment.design_spec.model_copy(deep=True)
    actual_design_spec.experiment_id = None
    actual_design_spec.arms[0].arm_id = None
    actual_design_spec.arms[1].arm_id = None
    assert actual_design_spec == base_request.design_spec
    assert created_experiment.power_analyses == base_request.power_analyses

    experiment_id = created_experiment.design_spec.experiment_id
    (arm1_id, arm2_id) = [arm.arm_id for arm in created_experiment.design_spec.arms]

    # Verify database state using the ids in the returned DesignSpec.
    experiment = xngin_session.scalars(
        select(tables.Experiment).where(tables.Experiment.id == experiment_id)
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == datasource_id
    assert experiment.experiment_type == "preassigned"
    assert experiment.participant_type == "test_participant_type"
    assert experiment.name == base_request.design_spec.experiment_name
    assert experiment.description == base_request.design_spec.description
    assert conftest.dates_equal(
        experiment.start_date, base_request.design_spec.start_date
    )
    assert conftest.dates_equal(experiment.end_date, base_request.design_spec.end_date)
    # Verify assignments were created
    assignments = xngin_session.scalars(
        select(tables.ArmAssignment).where(
            tables.ArmAssignment.experiment_id == experiment_id
        )
    ).all()
    assert len(assignments) == 100, {
        e.name: getattr(experiment, e.name) for e in tables.Experiment.__table__.columns
    }

    # Check one assignment to see if it looks roughly right
    sample_assignment: tables.ArmAssignment = assignments[0]
    assert sample_assignment.participant_type == "test_participant_type"
    assert sample_assignment.experiment_id == experiment_id
    assert sample_assignment.arm_id in {arm1_id, arm2_id}
    for stratum in sample_assignment.strata:
        assert stratum["field_name"] in {"current_income", "gender"}

    # Check for approximate balance in arm assignment
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 5  # Allow some wiggle room


def test_create_online_experiment_using_inline_schema_ds(
    testing_datasource_with_user_added, use_deterministic_random
):
    datasource_id = testing_datasource_with_user_added.ds.id
    base_request_json = make_createexperimentrequest_json(
        "test_participant_type", "online"
    )
    base_request = CreateExperimentRequest.model_validate(base_request_json)

    response = ppost(
        f"/v1/m/datasources/{datasource_id}/experiments",
        params={"random_state": 42},
        json=base_request_json,
    )
    assert response.status_code == 200, response.content
    created_experiment = CreateExperimentResponse.model_validate(response.json())
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
    assert assign_summary.balance_check is None
    assert assign_summary.sample_size == 0
    assert assign_summary.arm_sizes is not None
    assert all(a.size == 0 for a in assign_summary.arm_sizes)
    assert created_experiment.power_analyses is None
    # Check if the representations are equivalent
    # scrub the ids from the config for comparison
    actual_design_spec = created_experiment.design_spec.model_copy(deep=True)
    actual_design_spec.experiment_id = None
    actual_design_spec.arms[0].arm_id = None
    actual_design_spec.arms[1].arm_id = None
    assert actual_design_spec == base_request.design_spec


def test_get_experiment_assignment_for_preassigned_participant(testing_experiment):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id
    assignments = testing_experiment.arm_assignments

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/unassigned_id"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "unassigned_id"
    assert assignment_response.assignment is None

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/{assignments[0].participant_id}"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == assignments[0].participant_id
    assert assignment_response.assignment is not None


def test_get_experiment_assignment_for_online_participant(
    xngin_session, testing_datasource_with_user_added
):
    testing_experiment = make_experiment_and_arms(
        xngin_session, testing_datasource_with_user_added.ds, "online"
    )
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    # Create a new participant assignment.
    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id"
    )
    assert response.status_code == 200, response.content
    assignment_response = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response.experiment_id == experiment_id
    assert assignment_response.participant_id == "new_id"
    assert assignment_response.assignment is not None
    assert str(assignment_response.assignment.arm_id) in {
        arm.id for arm in testing_experiment.arms
    }

    # Get back the same assignment.
    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/assignments/new_id"
    )
    assert response.status_code == 200, response.content
    assignment_response2 = GetParticipantAssignmentResponse.model_validate(
        response.json()
    )
    assert assignment_response2 == assignment_response

    # Make sure there's only one db entry.
    assignment = xngin_session.scalars(
        select(tables.ArmAssignment).where(
            tables.ArmAssignment.experiment_id == experiment_id
        )
    ).one()
    assert assignment.participant_id == "new_id"
    assert assignment.arm_id == str(assignment_response.assignment.arm_id)


def test_experiments_analyze(testing_experiment):
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze"
    )

    assert response.status_code == 200, response.content
    experiment_analysis = ExperimentAnalysis.model_validate(response.json())
    assert experiment_analysis.experiment_id == experiment_id
    assert len(experiment_analysis.metric_analyses) == 1
    # Verify that only the first arm is marked as baseline by default
    metric_analysis = experiment_analysis.metric_analyses[0]
    baseline_arms = [arm for arm in metric_analysis.arm_analyses if arm.is_baseline]
    assert len(baseline_arms) == 1
    assert baseline_arms[0].is_baseline
    for analysis in experiment_analysis.metric_analyses:
        # Verify arm_ids match the database model.
        assert {arm.arm_id for arm in analysis.arm_analyses} == {
            arm.id for arm in testing_experiment.arms
        }


def test_experiments_analyze_for_experiment_with_no_participants(
    xngin_session, testing_datasource_with_user_added
):
    testing_experiment = make_experiment_and_arms(
        xngin_session, testing_datasource_with_user_added.ds, "online"
    )
    datasource_id = testing_experiment.datasource_id
    experiment_id = testing_experiment.id

    response = pget(
        f"/v1/m/datasources/{datasource_id}/experiments/{experiment_id}/analyze"
    )
    assert response.status_code == 422, response.content
    assert response.json()["message"] == "No participants found for experiment."


@pytest.mark.parametrize(
    "endpoint,initial_state,expected_status,expected_detail",
    [
        ("commit", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("commit", ExperimentState.COMMITTED, 304, None),  # No-op
        ("commit", ExperimentState.DESIGNING, 400, "Invalid state: designing"),
        ("commit", ExperimentState.ABORTED, 400, "Invalid state: aborted"),
        ("abandon", ExperimentState.DESIGNING, 204, None),  # Success case
        ("abandon", ExperimentState.ASSIGNED, 204, None),  # Success case
        ("abandon", ExperimentState.ABANDONED, 304, None),  # No-op
        ("abandon", ExperimentState.COMMITTED, 400, "Invalid state: committed"),
    ],
)
def test_admin_experiment_state_setting(
    xngin_session,
    testing_datasource_with_user_added,
    endpoint,
    initial_state,
    expected_status,
    expected_detail,
):
    # Initialize our state with an existing experiment who's state we want to modify.
    datasource_id = testing_datasource_with_user_added.ds.id
    experiment = make_insertable_experiment(initial_state, datasource_id=datasource_id)
    xngin_session.add(experiment)
    xngin_session.commit()

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
        xngin_session.refresh(experiment)
        assert experiment.state == expected_state
    # If failure case, verify the error message
    if expected_detail:
        assert response.json()["detail"] == expected_detail
