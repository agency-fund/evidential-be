"""Utilities for boostrapping entities in our app db."""

import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.routers.admin import admin_common
from xngin.apiserver.routers.admin.admin_api_types import AddWebhookToOrganizationRequest
from xngin.apiserver.routers.common_api_types import (
    Arm,
    ArmBandit,
    CMABExperimentSpec,
    Context,
    CreateExperimentRequest,
    DesignSpec,
    DesignSpecMetricRequest,
    Filter,
    MABExperimentSpec,
    OnlineFrequentistExperimentSpec,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import ContextType, LikelihoodTypes, PriorTypes, Relation
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.settings import Dsn, RemoteDatabaseConfig
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF

DEFAULT_ORGANIZATION_NAME = "My Organization"
TESTING_DWH_DATASOURCE_NAME = "Local DWH"
ALT_TESTING_DWH_DATASOURCE_NAME = "Alternate Local DWH"


async def _create_and_commit_experiment(session: AsyncSession, datasource: tables.Datasource, design_spec: DesignSpec):
    result = await experiments_common.create_experiment_impl(
        CreateExperimentRequest(design_spec=design_spec),
        datasource,
        session,
        chosen_n=100,
        stratify_on_metrics=False,
        random_state=None,
        validated_webhooks=[],
    )
    experiment = await session.get_one(tables.Experiment, result.experiment_id)
    await experiments_common.commit_experiment_impl(session, experiment)


async def _maybe_create_developer_samples(
    session: AsyncSession, organization: tables.Organization, testing_dwh_dsn: str | None
):
    if not testing_dwh_dsn:
        return

    from xngin.apiserver.testing.wide_dwh_def import WIDE_DWH_PARTICIPANT_DEF  # noqa: PLC0415

    _ = admin_common.create_webhook_impl(
        session,
        organization.id,
        AddWebhookToOrganizationRequest(type="experiment.created", name="Sample Webhook", url="http://localhost:8000"),
    )

    datasource = await admin_common.create_datasource_impl(
        session,
        organization,
        TESTING_DWH_DATASOURCE_NAME,
        RemoteDatabaseConfig(
            participants=[TESTING_DWH_PARTICIPANT_DEF],
            type="remote",
            dwh=Dsn.from_url(testing_dwh_dsn),
        ),
    )

    alt_datasource = await admin_common.create_datasource_impl(
        session,
        organization,
        ALT_TESTING_DWH_DATASOURCE_NAME,
        RemoteDatabaseConfig(
            participants=[WIDE_DWH_PARTICIPANT_DEF],
            type="remote",
            dwh=Dsn.from_url(testing_dwh_dsn),
        ),
    )
    await session.flush()

    await _create_and_commit_experiment(
        session,
        datasource,
        PreassignedFrequentistExperimentSpec(
            participant_type=TESTING_DWH_PARTICIPANT_DEF.participant_type,
            experiment_name="Preassigned",
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
    )

    await _create_and_commit_experiment(
        session,
        datasource,
        OnlineFrequentistExperimentSpec(
            participant_type=TESTING_DWH_PARTICIPANT_DEF.participant_type,
            experiment_name="Online",
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
    )

    await _create_and_commit_experiment(
        session,
        datasource,
        MABExperimentSpec(
            participant_type="testing_dwh_participant",
            experiment_name="MAB",
            description="Hypothesis",
            start_date=datetime.datetime.now() - datetime.timedelta(days=7),
            end_date=datetime.datetime.now() + datetime.timedelta(days=7),
            prior_type=PriorTypes.BETA,
            reward_type=LikelihoodTypes.BERNOULLI,
            arms=[
                ArmBandit(arm_name="Control", arm_description="First arm", alpha_init=1.0, beta_init=1.0),
                ArmBandit(arm_name="Treatment", arm_description="Second arm", alpha_init=2.0, beta_init=2.0),
            ],
        ),
    )

    await _create_and_commit_experiment(
        session,
        datasource,
        CMABExperimentSpec(
            participant_type="testing_dwh_participant",
            experiment_name="CMAB",
            description="Hypothesis",
            start_date=datetime.datetime.now() - datetime.timedelta(days=7),
            end_date=datetime.datetime.now() + datetime.timedelta(days=7),
            prior_type=PriorTypes.NORMAL,
            reward_type=LikelihoodTypes.NORMAL,
            arms=[
                ArmBandit(arm_name="Control", arm_description="First arm", mu_init=0.0, sigma_init=1.0),
                ArmBandit(arm_name="Treatment", arm_description="Second arm", mu_init=1.0, sigma_init=2.0),
            ],
            contexts=[
                Context(
                    context_name="age", context_description="Age of participant", value_type=ContextType.REAL_VALUED
                ),
                Context(
                    context_name="gender",
                    context_description="Gender of participant",
                    value_type=ContextType.BINARY,
                ),
            ],
        ),
    )

    await _create_and_commit_experiment(
        session,
        alt_datasource,
        PreassignedFrequentistExperimentSpec(
            participant_type=WIDE_DWH_PARTICIPANT_DEF.participant_type,
            experiment_name="Wide Preassigned",
            description="Hypothesis",
            start_date=datetime.datetime.now() - datetime.timedelta(days=7),
            end_date=datetime.datetime.now() + datetime.timedelta(days=7),
            arms=[
                Arm(arm_name="Control", arm_description="First arm"),
                Arm(arm_name="Treatment", arm_description="Second arm"),
            ],
            filters=[Filter(field_name="household_income", relation=Relation.BETWEEN, value=[100, None])],
            strata=[],
            metrics=[DesignSpecMetricRequest(field_name="savings_balance", metric_pct_change=0.10)],
        ),
    )


async def create_entities_for_first_time_user(
    session: AsyncSession, user: tables.User, testing_dwh_dsn: str | None
) -> tables.User:
    """Bootstraps a user with organization, datasources, and optionally experiments.

    When testing_dwh_dsn is provided, we assume that the user is a developer working on a development instance with an
    accessible instance of a testing DWH. New users created in these environments will have experiments and a testing
    datasource corresponding to the testing DWH created.

    When testing_dwh_dsn is None or empty, we create only the minimum entities necessary for the application to
    function: a NoDWH datasource, and an Organization. This is the standard production deployment configuration.
    """
    organization = await admin_common.create_organization_impl(session, user, DEFAULT_ORGANIZATION_NAME)
    await session.flush()
    await _maybe_create_developer_samples(session, organization, testing_dwh_dsn)
    return user
