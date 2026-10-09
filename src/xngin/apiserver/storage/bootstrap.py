"""Utilities for boostrapping entities in our app db."""

import datetime

from sqlalchemy import update
from sqlalchemy.orm import Session

from xngin.apiserver.routers.admin import admin_common
from xngin.apiserver.routers.admin.admin_api_types import AddExperimentCreatedWebhookRequest
from xngin.apiserver.routers.common_api_types import (
    Arm,
    ArmBandit,
    CMABExperimentSpec,
    Context,
    CreateExperimentRequest,
    DesignSpecMetricRequest,
    Filter,
    MABDwhExperimentSpec,
    MABExperimentSpec,
    OnlineFrequentistExperimentSpec,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import ContextType, LikelihoodTypes, PriorTypes, Relation
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.settings import Dsn, RemoteDatabaseConfig
from xngin.apiserver.snapshots.fake_data import (
    LATE_BREAKOUT,
    STEADY_GAIN,
    seed_historical_snapshots,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_TABLE_NAME
from xngin.apiserver.testing.wide_dwh_def import WIDE_DWH_TABLE_NAME

DEFAULT_ORGANIZATION_NAME = "My Organization"
TESTING_DWH_DATASOURCE_NAME = "Local DWH"
ALT_TESTING_DWH_DATASOURCE_NAME = "Alternate Local DWH"


def _create_and_commit_experiment(
    session: Session,
    datasource: tables.Datasource,
    create_experiment_request: CreateExperimentRequest,
):
    result = experiments_common.create_experiment_impl(
        create_experiment_request,
        datasource,
        session,
        stratify_on_metrics=False,
        random_state=None,
        validated_webhooks=[],
    )
    experiment = session.get_one(tables.Experiment, result.experiment_id)
    experiments_common.commit_experiment_impl(session, experiment)
    return experiment


def _create_draws(
    session: Session,
    experiment: tables.Experiment,
    participant_ids: list[str],
    *,
    created_at: datetime.datetime | None = None,
):
    """Assigns each participant to an arm. created_at backdates the draws, e.g. past the autofail window."""
    for i, participant_id in enumerate(participant_ids):
        experiments_common.create_assignment_for_participant(session, experiment, participant_id, random_state=i)
    if created_at is not None:
        session.execute(
            update(tables.Draw)
            .where(tables.Draw.experiment_id == experiment.id, tables.Draw.participant_id.in_(participant_ids))
            .values(created_at=created_at)
        )


def _maybe_create_developer_samples(session: Session, organization: tables.Organization, testing_dwh_dsn: str | None):
    if not testing_dwh_dsn:
        return

    _ = admin_common.create_webhook_impl(
        session,
        organization.id,
        AddExperimentCreatedWebhookRequest(
            name="Sample Webhook", url="http://localhost:8000", type="experiment.created", direction="outbound"
        ),
    )

    datasource = admin_common.create_datasource_impl(
        session,
        organization,
        TESTING_DWH_DATASOURCE_NAME,
        RemoteDatabaseConfig(
            type="remote",
            dwh=Dsn.from_url(testing_dwh_dsn),
        ),
    )

    alt_datasource = admin_common.create_datasource_impl(
        session,
        organization,
        ALT_TESTING_DWH_DATASOURCE_NAME,
        RemoteDatabaseConfig(
            type="remote",
            dwh=Dsn.from_url(testing_dwh_dsn),
        ),
    )
    session.flush()

    now = datetime.datetime.now(datetime.UTC)
    preassigned = _create_and_commit_experiment(
        session,
        datasource,
        CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_name="Preassigned - steady gain",
                description="Hypothesis",
                table_name=TESTING_DWH_TABLE_NAME,
                primary_key="id",
                start_date=now - datetime.timedelta(days=7),
                end_date=now + datetime.timedelta(days=7),
                arms=[
                    Arm(arm_name="Control", arm_description="First arm"),
                    Arm(arm_name="Treatment", arm_description="Second arm"),
                ],
                filters=[Filter(field_name="baseline_income", relation=Relation.BETWEEN, value=[100, None])],
                strata=[],
                metrics=[DesignSpecMetricRequest(field_name="current_income", metric_pct_change=0.10)],
                desired_n=100,
            ),
        ),
    )
    seed_historical_snapshots(session, preassigned, STEADY_GAIN)

    preassigned_2 = _create_and_commit_experiment(
        session,
        datasource,
        CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_name="Preassigned - late breakout",
                description="Hypothesis",
                table_name=TESTING_DWH_TABLE_NAME,
                primary_key="id",
                start_date=now - datetime.timedelta(days=7),
                end_date=now + datetime.timedelta(days=7),
                arms=[
                    Arm(arm_name="Control", arm_description="First arm"),
                    Arm(arm_name="Treatment", arm_description="Second arm"),
                ],
                filters=[Filter(field_name="baseline_income", relation=Relation.BETWEEN, value=[100, None])],
                strata=[],
                metrics=[DesignSpecMetricRequest(field_name="current_income", metric_pct_change=0.10)],
                desired_n=100,
            ),
        ),
    )
    seed_historical_snapshots(session, preassigned_2, LATE_BREAKOUT)

    _create_and_commit_experiment(
        session,
        datasource,
        CreateExperimentRequest(
            design_spec=OnlineFrequentistExperimentSpec(
                experiment_name="Online",
                description="Hypothesis",
                table_name=TESTING_DWH_TABLE_NAME,
                primary_key="id",
                start_date=now - datetime.timedelta(days=7),
                end_date=now + datetime.timedelta(days=7),
                arms=[
                    Arm(arm_name="Control", arm_description="First arm"),
                    Arm(arm_name="Treatment", arm_description="Second arm"),
                ],
                filters=[],
                strata=[],
                metrics=[DesignSpecMetricRequest(field_name="current_income", metric_pct_change=0.10)],
            ),
        ),
    )

    mab = _create_and_commit_experiment(
        session,
        datasource,
        CreateExperimentRequest(
            design_spec=MABExperimentSpec(
                experiment_name="MAB",
                description="Hypothesis",
                start_date=now - datetime.timedelta(days=7),
                end_date=now + datetime.timedelta(days=7),
                prior_type=PriorTypes.BETA,
                reward_type=LikelihoodTypes.BERNOULLI,
                enable_autofail=True,
                autofail_window=24,
                autofail_outcome_value=0.0,
                arms=[
                    ArmBandit(arm_name="Control", arm_description="First arm", alpha_init=1.0, beta_init=1.0),
                    ArmBandit(arm_name="Treatment", arm_description="Second arm", alpha_init=2.0, beta_init=2.0),
                ],
            )
        ),
    )
    # Draws past the autofail window are resolved by the first autofail run; the fresh ones wait out the window.
    _create_draws(
        session,
        mab,
        ["stale-1", "stale-2", "stale-3"],
        created_at=now - datetime.timedelta(days=2),
    )
    _create_draws(session, mab, ["fresh-1", "fresh-2"])

    _create_and_commit_experiment(
        session,
        datasource,
        CreateExperimentRequest(
            design_spec=CMABExperimentSpec(
                experiment_name="CMAB",
                description="Hypothesis",
                start_date=now - datetime.timedelta(days=7),
                end_date=now + datetime.timedelta(days=7),
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
            )
        ),
    )

    _create_and_commit_experiment(
        session,
        alt_datasource,
        CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_name="Preassigned - wide",
                description="Hypothesis",
                table_name=WIDE_DWH_TABLE_NAME,
                primary_key="id",
                start_date=now - datetime.timedelta(days=7),
                end_date=now + datetime.timedelta(days=7),
                arms=[
                    Arm(arm_name="Control", arm_description="First arm"),
                    Arm(arm_name="Treatment", arm_description="Second arm"),
                ],
                filters=[Filter(field_name="household_income", relation=Relation.BETWEEN, value=[100, None])],
                strata=[],
                metrics=[DesignSpecMetricRequest(field_name="savings_balance", metric_pct_change=0.10)],
                desired_n=100,
            ),
        ),
    )

    mab_dwh = _create_and_commit_experiment(
        session,
        alt_datasource,
        CreateExperimentRequest(
            design_spec=MABDwhExperimentSpec(
                experiment_name="MAB - wide DWH",
                description="Hypothesis",
                table_name=WIDE_DWH_TABLE_NAME,
                primary_key="id",
                target_field_name="converted",
                start_date=now - datetime.timedelta(days=7),
                end_date=now + datetime.timedelta(days=7),
                prior_type=PriorTypes.BETA,
                reward_type=LikelihoodTypes.BERNOULLI,
                arms=[
                    ArmBandit(arm_name="Control", arm_description="First arm", alpha_init=1.0, beta_init=1.0),
                    ArmBandit(arm_name="Treatment", arm_description="Second arm", alpha_init=2.0, beta_init=2.0),
                ],
            )
        ),
    )
    # The first DWH pull ingests outcomes for participants 1-10, whose converted column is set in the wide DWH.
    # Participant 61's converted column is NULL, so its draw stays pending across pulls.
    _create_draws(session, mab_dwh, [str(i) for i in range(1, 11)] + ["61"])


def create_entities_for_first_time_user(
    session: Session, user: tables.User, testing_dwh_dsn: str | None
) -> tables.User:
    """Bootstraps a user with organization, datasources, and optionally experiments.

    When testing_dwh_dsn is provided, we assume that the user is a developer working on a development instance with an
    accessible instance of a testing DWH. New users created in these environments will have experiments and a testing
    datasource corresponding to the testing DWH created.

    When testing_dwh_dsn is None or empty, we create only the minimum entities necessary for the application to
    function: a NoDWH datasource, and an Organization. This is the standard production deployment configuration.
    """
    organization = admin_common.create_organization_impl(session, user, DEFAULT_ORGANIZATION_NAME)
    session.flush()
    _maybe_create_developer_samples(session, organization, testing_dwh_dsn)
    return user
