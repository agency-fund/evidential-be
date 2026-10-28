"""Utilities for boostrapping entities in our app db."""

import base64
import datetime as dt
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from sqlalchemy import make_url, update
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
    Stratum,
)
from xngin.apiserver.routers.common_enums import ContextType, LikelihoodTypes, PriorTypes, Relation
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.settings import BqDsn, Dsn, Dwh, GcpServiceAccountInfo, RemoteDatabaseConfig
from xngin.apiserver.snapshots.fake_data import (
    LATE_BREAKOUT,
    STEADY_GAIN,
    seed_historical_snapshots,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.testing import stress_dwh_def
from xngin.apiserver.testing.stress_dwh_def import ExperimentFields
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_TABLE_NAME
from xngin.apiserver.testing.wide_dwh_def import WIDE_DWH_TABLE_NAME

DEFAULT_ORGANIZATION_NAME = "My Organization"
TESTING_DWH_DATASOURCE_NAME = "Local DWH"
ALT_TESTING_DWH_DATASOURCE_NAME = "Alternate Local DWH"
PGSTRESS_DATASOURCE_NAME = "pgstress"
BQSTRESS_DATASOURCE_NAME = "bqstress"
RSSTRESS_DATASOURCE_NAME = "rsstress"


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
    created_at: dt.datetime | None = None,
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

    now = dt.datetime.now(dt.UTC)
    preassigned = _create_and_commit_experiment(
        session,
        datasource,
        CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_name="Preassigned - steady gain",
                description="Hypothesis",
                table_name=TESTING_DWH_TABLE_NAME,
                primary_key="id",
                start_date=now - dt.timedelta(days=7),
                end_date=now + dt.timedelta(days=7),
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
                start_date=now - dt.timedelta(days=7),
                end_date=now + dt.timedelta(days=7),
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
                start_date=now - dt.timedelta(days=7),
                end_date=now + dt.timedelta(days=7),
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
                start_date=now - dt.timedelta(days=7),
                end_date=now + dt.timedelta(days=7),
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
        created_at=now - dt.timedelta(days=2),
    )
    _create_draws(session, mab, ["fresh-1", "fresh-2"])

    _create_and_commit_experiment(
        session,
        datasource,
        CreateExperimentRequest(
            design_spec=CMABExperimentSpec(
                experiment_name="CMAB",
                description="Hypothesis",
                start_date=now - dt.timedelta(days=7),
                end_date=now + dt.timedelta(days=7),
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
                start_date=now - dt.timedelta(days=7),
                end_date=now + dt.timedelta(days=7),
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
                start_date=now - dt.timedelta(days=7),
                end_date=now + dt.timedelta(days=7),
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


def _bigquery_dwh(bigquery_dsn: str) -> BqDsn:
    """A BqDsn for bigquery://<project>/<dataset>?credentials_path=<key file>, carrying the key's content.

    The data source holds the key itself, as data sources created through the API do: the API server refuses to run
    with ambient Google credentials.
    """
    url = make_url(bigquery_dsn)
    match url.query.get("credentials_path"):
        case str(path):
            pass
        case _:
            raise ValueError("The BigQuery DSN must name its service account key file with ?credentials_path=.")
    return BqDsn(
        driver="bigquery",
        project_id=url.host,
        dataset_id=url.database,
        credentials=GcpServiceAccountInfo(content_base64=base64.standard_b64encode(Path(path).read_bytes()).decode()),
    )


def _create_stress_samples(
    session: Session, organization: tables.Organization, name: str, dwh: Dwh, table_name: str, fields: ExperimentFields
):
    """Adds a data source for a stress test table, with a preassigned experiment on it."""
    datasource = admin_common.create_datasource_impl(
        session, organization, name, RemoteDatabaseConfig(type="remote", dwh=dwh)
    )
    session.flush()
    now = dt.datetime.now(dt.UTC)
    _create_and_commit_experiment(
        session,
        datasource,
        CreateExperimentRequest(
            design_spec=PreassignedFrequentistExperimentSpec(
                experiment_name=f"Preassigned - {name}",
                description="Fields whose names need quoting",
                table_name=table_name,
                primary_key=fields.primary_key,
                start_date=now - dt.timedelta(days=7),
                end_date=now + dt.timedelta(days=7),
                arms=[
                    Arm(arm_name="Control", arm_description="First arm"),
                    Arm(arm_name="Treatment", arm_description="Second arm"),
                ],
                filters=[
                    Filter(
                        field_name=fields.filter_field, relation=Relation.INCLUDES, value=list(fields.filter_values)
                    ),
                    Filter(
                        field_name=fields.last_login,
                        relation=Relation.BETWEEN,
                        value=[stress_dwh_def.LOGGED_IN_SINCE, None],
                    ),
                ],
                strata=[Stratum(field_name=fields.strata)],
                metrics=[DesignSpecMetricRequest(field_name=fields.metric, metric_pct_change=0.10)],
                desired_n=40,
            ),
        ),
    )


def _create_stress_samples_or_warn(session: Session, name: str, create: Callable[[], None]) -> None:
    """Runs create, or undoes its changes and logs a warning if it fails.

    A warehouse that is misconfigured, unreachable or missing the stress test table must not block the dev
    environment's first sign-in.
    """
    try:
        with session.begin_nested():
            create()
    except Exception:
        logger.opt(exception=True).warning("Skipped the {} data source, which could not be created.", name)


def _maybe_create_stress_samples(
    session: Session,
    organization: tables.Organization,
    testing_dwh_dsn: str | None,
    bigquery_dsn: str | None,
    redshift_dsn: str | None,
):
    """Adds a data source for each configured warehouse that holds the stress test table."""
    if testing_dwh_dsn:
        dsn = testing_dwh_dsn
        _create_stress_samples_or_warn(
            session,
            PGSTRESS_DATASOURCE_NAME,
            lambda: _create_stress_samples(
                session,
                organization,
                PGSTRESS_DATASOURCE_NAME,
                Dsn.from_url(dsn).model_copy(update={"search_path": f'"{stress_dwh_def.POSTGRES.schema}"'}),
                stress_dwh_def.POSTGRES.table,
                stress_dwh_def.POSTGRES_EXPERIMENT,
            ),
        )
    if bigquery_dsn:
        bq_dsn = bigquery_dsn
        _create_stress_samples_or_warn(
            session,
            BQSTRESS_DATASOURCE_NAME,
            lambda: _create_stress_samples(
                session,
                organization,
                BQSTRESS_DATASOURCE_NAME,
                _bigquery_dwh(bq_dsn),
                stress_dwh_def.BIGQUERY.table,
                stress_dwh_def.BIGQUERY_EXPERIMENT,
            ),
        )
    if redshift_dsn:
        rs_dsn = redshift_dsn
        _create_stress_samples_or_warn(
            session,
            RSSTRESS_DATASOURCE_NAME,
            lambda: _create_stress_samples(
                session,
                organization,
                RSSTRESS_DATASOURCE_NAME,
                Dsn.from_url(rs_dsn).model_copy(update={"search_path": stress_dwh_def.REDSHIFT_SEARCH_PATH}),
                stress_dwh_def.REDSHIFT.table.lower(),  # Redshift lists the table in lowercase.
                stress_dwh_def.REDSHIFT_EXPERIMENT,
            ),
        )


def create_entities_for_first_time_user(
    session: Session,
    user: tables.User,
    testing_dwh_dsn: str | None,
    *,
    bigquery_dsn: str | None = None,
    redshift_dsn: str | None = None,
) -> tables.User:
    """Bootstraps a user with organization, datasources, and optionally experiments.

    When testing_dwh_dsn is provided, we assume that the user is a developer working on a development instance with an
    accessible instance of a testing DWH. New users created in these environments will have experiments and a testing
    datasource corresponding to the testing DWH created.

    When testing_dwh_dsn is None or empty, we create only the minimum entities necessary for the application to
    function: a NoDWH datasource, and an Organization. This is the standard production deployment configuration.

    When bigquery_dsn or redshift_dsn is provided, it names a warehouse holding the stress test table, which gets a data
    source as well.
    """
    organization = admin_common.create_organization_impl(session, user, DEFAULT_ORGANIZATION_NAME)
    session.flush()
    _maybe_create_developer_samples(session, organization, testing_dwh_dsn)
    _maybe_create_stress_samples(session, organization, testing_dwh_dsn, bigquery_dsn, redshift_dsn)
    return user
