"""Utilities for boostrapping entities in our app db."""

import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.routers.admin import admin_common
from xngin.apiserver.routers.common_api_types import (
    Arm,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    Filter,
    OnlineFrequentistExperimentSpec,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import Relation
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.settings import Dsn, RemoteDatabaseConfig
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF

DEFAULT_ORGANIZATION_NAME = "My Organization"
DEFAULT_DWH_SOURCE_NAME = "Local DWH"


async def _maybe_create_developer_samples(
    session: AsyncSession, organization: tables.Organization, testing_dwh_dsn: str | None
):
    if testing_dwh_dsn is None:
        return

    datasource = await admin_common.create_datasource_impl(
        session,
        organization,
        DEFAULT_DWH_SOURCE_NAME,
        RemoteDatabaseConfig(
            participants=[TESTING_DWH_PARTICIPANT_DEF],
            type="remote",
            dwh=Dsn.from_url(testing_dwh_dsn),
        ),
    )
    await session.flush()
    datasource_id = await datasource.awaitable_attrs.id
    create_preassigned_experiment = await experiments_common.create_experiment_impl(
        CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                participant_type=TESTING_DWH_PARTICIPANT_DEF.participant_type,
                experiment_name="My Preassigned Experiment",
                description="Hypothesis",
                start_date=datetime.datetime.now() - datetime.timedelta(days=7),
                end_date=datetime.datetime.now() + datetime.timedelta(days=7),
                arms=[
                    Arm(arm_name="Control", arm_description="First arm"),
                    Arm(arm_name="Treatment", arm_description="Second arm"),
                ],
                filters=[Filter(field_name="baseline_income", relation=Relation.BETWEEN, value=[100, None])],
                strata=[],
                metrics=[DesignSpecMetricRequest(field_name="current_income", metric_pct_change=0.10)],
            ),
        ),
        await session.get_one(tables.Datasource, datasource_id),
        session,
        chosen_n=100,
        stratify_on_metrics=False,
        random_state=None,
        validated_webhooks=[],
    )
    preassigned_experiment = await session.get_one(tables.Experiment, create_preassigned_experiment.experiment_id)
    await experiments_common.commit_experiment_impl(session, preassigned_experiment)

    create_online_experiment = await experiments_common.create_experiment_impl(
        CreateExperimentRequest(
            design_spec=OnlineFrequentistExperimentSpec(
                participant_type=TESTING_DWH_PARTICIPANT_DEF.participant_type,
                experiment_name="My Online Experiment",
                description="Hypothesis",
                start_date=datetime.datetime.now() - datetime.timedelta(days=7),
                end_date=datetime.datetime.now() + datetime.timedelta(days=7),
                arms=[
                    Arm(arm_name="Control", arm_description="First arm"),
                    Arm(arm_name="Treatment", arm_description="Second arm"),
                ],
                filters=[],
                strata=[],
                metrics=[DesignSpecMetricRequest(field_name="current_income", metric_pct_change=0.10)],
            ),
        ),
        await session.get_one(tables.Datasource, datasource_id),
        session,
        chosen_n=100,
        stratify_on_metrics=False,
        random_state=None,
        validated_webhooks=[],
    )
    online_experiment = await session.get_one(tables.Experiment, create_online_experiment.experiment_id)
    await experiments_common.commit_experiment_impl(session, online_experiment)


async def setup_user_and_first_datasource(
    session: AsyncSession, user: tables.User, testing_dwh_dsn: str | None
) -> tables.User:
    """Adds models to User such that they can have a good first time experience with the application.

    Users will have an organization and a NoDWH datasource created for them.

    If testing_dwh_dsn is provided, a datasource and a participant type for that DWH will be created. testing_dwh_dsn
    must refer to a testing dwh instance. This is usually only used in development environments.
    """
    organization = await admin_common.create_organization_impl(session, user, DEFAULT_ORGANIZATION_NAME)
    await session.flush()
    await _maybe_create_developer_samples(session, organization, testing_dwh_dsn)
    return user
