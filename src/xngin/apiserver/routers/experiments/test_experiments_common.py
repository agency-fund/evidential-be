import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
import pytest
from deepdiff import DeepDiff
from fastapi import HTTPException
from numpy.random import MT19937, RandomState
from pydantic import TypeAdapter
from sqlalchemy import Boolean, Column, MetaData, String, Table, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.schema import CreateTable

from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import (
    CMABExperimentSpec,
    CreateExperimentRequest,
    DesignSpec,
    DesignSpecMetric,
    ExperimentsType,
    LikelihoodTypes,
    MABExperimentSpec,
    MetricPowerAnalysis,
    MetricType,
    OnlineFrequentistExperimentSpec,
    PowerResponse,
    PreassignedFrequentistExperimentSpec,
    PriorTypes,
    Stratum,
)
from xngin.apiserver.routers.common_enums import ExperimentState, StopAssignmentReason
from xngin.apiserver.routers.experiments.experiments_common import (
    ExperimentsAssignmentError,
    abandon_experiment_impl,
    commit_experiment_impl,
    create_assignment_for_participant,
    create_bandit_online_experiment_impl,
    create_dwh_experiment_impl,
    create_preassigned_experiment_impl,
    experiment_assignments_to_csv_generator,
    get_assign_summary,
    get_existing_assignment_for_participant,
    get_experiment_assignments_impl,
    list_organization_or_datasource_experiments_impl,
    update_bandit_arm_with_outcome_impl,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.testing.assertions import assert_dates_equal


def make_createexperimentrequest_json(
    participant_type: str = "test_participant_type",
    experiment_type: str = "freq_preassigned",
    with_ids: bool = False,
    prior_type: PriorTypes = PriorTypes.NORMAL,
    reward_type: LikelihoodTypes = LikelihoodTypes.NORMAL,
):
    """Make a basic CreateExperimentRequest JSON object.

    This does not add any power analyses or balance checks, nor do any validation.
    """
    experiment_id = tables.experiment_id_factory() if with_ids else None
    arm1_id = tables.arm_id_factory() if with_ids else None
    arm2_id = tables.arm_id_factory() if with_ids else None

    experiment_type = ExperimentsType(experiment_type)
    match experiment_type:
        case ExperimentsType.FREQ_PREASSIGNED | ExperimentsType.FREQ_ONLINE:
            return {
                "design_spec": {
                    **({"experiment_id": experiment_id} if experiment_id is not None else {}),
                    "participant_type": participant_type,
                    "experiment_name": "test",
                    "description": "test",
                    "experiment_type": experiment_type,
                    # Attach UTC tz, but use dates_equal() to compare to respect db storage support
                    "start_date": "2024-01-01T00:00:00+00:00",
                    # default our experiment to end in the future
                    "end_date": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                    "arms": [
                        {
                            **({"arm_id": arm1_id} if arm1_id is not None else {}),
                            "arm_name": "control",
                            "arm_description": "control",
                        },
                        {
                            **({"arm_id": arm2_id} if arm2_id is not None else {}),
                            "arm_name": "treatment",
                            "arm_description": "treatment",
                        },
                    ],
                    "filters": [],
                    "strata": [{"field_name": "gender"}],
                    "metrics": [
                        {
                            "field_name": "is_onboarded",
                            "metric_pct_change": 0.1,
                        }
                    ],
                    "power": 0.8,
                    "alpha": 0.05,
                    "fstat_thresh": 0.2,
                }
            }
        case ExperimentsType.MAB_ONLINE:
            return {
                "design_spec": {
                    **({"experiment_id": experiment_id} if experiment_id is not None else {}),
                    "participant_type": participant_type,
                    "experiment_name": "test",
                    "description": "test",
                    # Attach UTC tz, but use dates_equal() to compare to respect db storage support
                    "start_date": "2024-01-01T00:00:00+00:00",
                    # default our experiment to end in the future
                    "end_date": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                    "experiment_type": "mab_online",
                    "prior_type": prior_type,
                    "reward_type": reward_type,
                    "arms": [
                        {
                            **({"arm_id": arm1_id} if arm1_id is not None else {}),
                            "arm_name": "string",
                            "arm_description": "string",
                            "alpha_init": 50.0 if prior_type == PriorTypes.BETA else None,
                            "beta_init": 1.0 if prior_type == PriorTypes.BETA else None,
                            "mu_init": 10.0 if prior_type == PriorTypes.NORMAL else None,
                            "sigma_init": 1.0 if prior_type == PriorTypes.NORMAL else None,
                        },
                        {
                            **({"arm_id": arm2_id} if arm2_id is not None else {}),
                            "arm_name": "string",
                            "arm_description": "string",
                            "alpha_init": 1.0 if prior_type == PriorTypes.BETA else None,
                            "beta_init": 50.0 if prior_type == PriorTypes.BETA else None,
                            "mu_init": -10.0 if prior_type == PriorTypes.NORMAL else None,
                            "sigma_init": 1.0 if prior_type == PriorTypes.NORMAL else None,
                        },
                    ],
                }
            }
        case ExperimentsType.CMAB_ONLINE:
            context1_id = tables.context_id_factory() if with_ids else None
            context2_id = tables.context_id_factory() if with_ids else None
            return {
                "design_spec": {
                    **({"experiment_id": experiment_id} if experiment_id is not None else {}),
                    "participant_type": "string",
                    "experiment_name": "test",
                    "description": "test",
                    # Attach UTC tz, but use dates_equal() to compare to respect db storage support
                    "start_date": "2024-01-01T00:00:00+00:00",
                    # default our experiment to end in the future
                    "end_date": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                    "experiment_type": "cmab_online",
                    "prior_type": prior_type,
                    "reward_type": reward_type,
                    "arms": [
                        {
                            **({"arm_id": arm1_id} if arm1_id is not None else {}),
                            "arm_name": "Arm 1",
                            "arm_description": "Arm 1",
                            "mu_init": 10.0,
                            "sigma_init": 1.0,
                        },
                        {
                            **({"arm_id": arm2_id} if arm2_id is not None else {}),
                            "arm_name": "Arm 2",
                            "arm_description": "Arm 2",
                            "mu_init": -10.0,
                            "sigma_init": 1.0,
                        },
                    ],
                    "contexts": [
                        {
                            **({"context_id": context1_id} if context1_id is not None else {}),
                            "context_name": "Context 1",
                            "context_description": "Context 1",
                            "value_type": "binary",
                        },
                        {
                            **({"context_id": context2_id} if context2_id is not None else {}),
                            "context_name": "Context 2",
                            "context_description": "Context 2",
                            "value_type": "real-valued",
                        },
                    ],
                }
            }
        case _:
            raise ValueError(f"Invalid experiment type: {experiment_type}")


def make_create_preassigned_experiment_request(
    with_ids: bool = False,
) -> CreateExperimentRequest:
    request = make_createexperimentrequest_json(with_ids=with_ids, experiment_type=ExperimentsType.FREQ_PREASSIGNED)
    return TypeAdapter(CreateExperimentRequest).validate_python(request)


def make_create_online_experiment_request(
    with_ids: bool = False,
) -> CreateExperimentRequest:
    request = make_createexperimentrequest_json(with_ids=with_ids, experiment_type=ExperimentsType.FREQ_ONLINE)
    return TypeAdapter(CreateExperimentRequest).validate_python(request)


def make_create_online_bandit_experiment_request(
    experiment_type: ExperimentsType = ExperimentsType.MAB_ONLINE,
    with_ids: bool = False,
    reward_type: LikelihoodTypes = LikelihoodTypes.NORMAL,
    prior_type: PriorTypes = PriorTypes.NORMAL,
) -> CreateExperimentRequest:
    request = make_createexperimentrequest_json(
        with_ids=with_ids, experiment_type=experiment_type, prior_type=prior_type, reward_type=reward_type
    )
    return TypeAdapter(CreateExperimentRequest).validate_python(request)


def make_insertable_experiment(
    datasource: tables.Datasource,
    state: ExperimentState = ExperimentState.COMMITTED,
    experiment_type: ExperimentsType = ExperimentsType.FREQ_PREASSIGNED,
    with_ids: bool = True,
    prior_type: PriorTypes = PriorTypes.NORMAL,
    reward_type: LikelihoodTypes = LikelihoodTypes.NORMAL,
) -> tuple[tables.Experiment, DesignSpec]:
    """Make a minimal experiment with arms ready for insertion into the database for tests.

    This does not add any power analyses or balance checks.
    """
    request = make_createexperimentrequest_json(
        experiment_type=experiment_type, with_ids=with_ids, prior_type=prior_type, reward_type=reward_type
    )
    design_spec: DesignSpec = TypeAdapter(DesignSpec).validate_python(request["design_spec"])
    stopped_assignments_at: datetime | None = None
    stopped_assignments_reason: StopAssignmentReason | None = None
    if experiment_type == ExperimentsType.FREQ_PREASSIGNED:
        stopped_assignments_at = datetime.now(UTC)
        stopped_assignments_reason = StopAssignmentReason.PREASSIGNED

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource.id,
        organization_id=datasource.organization_id,
        experiment_type=experiment_type,
        design_spec=design_spec,
        state=state,
        stopped_assignments_at=stopped_assignments_at,
        stopped_assignments_reason=stopped_assignments_reason,
    )
    experiment = experiment_converter.get_experiment()
    return experiment, experiment_converter.get_design_spec()


async def insert_experiment_and_arms(
    xngin_session: AsyncSession,
    datasource: tables.Datasource,
    experiment_type: ExperimentsType = ExperimentsType.FREQ_PREASSIGNED,
    state=ExperimentState.COMMITTED,
    end_date: datetime | None = None,
    prior_type: PriorTypes = PriorTypes.NORMAL,
    reward_type: LikelihoodTypes = LikelihoodTypes.NORMAL,
):
    """Creates an experiment and arms and commits them to the database.

    Returns the new ORM experiment object.
    """
    experiment, _ = make_insertable_experiment(
        datasource=datasource,
        state=state,
        experiment_type=experiment_type,
        prior_type=prior_type,
        reward_type=reward_type,
    )
    # Override the end date if provided.
    if end_date is not None:
        experiment.end_date = end_date
    xngin_session.add(experiment)
    await xngin_session.commit()
    return experiment


@dataclass
class MockRow:
    """Simulate the bits of a sqlalchemy Row that we need here."""

    participant_id: str
    gender: str
    is_onboarded: bool
    region: str = "North"  # Default value for backward compatibility

    def _asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@pytest.fixture
def sample_table():
    """Create a mock SQLAlchemy table that works with make_create_preassigned_experiment_request()"""
    metadata_obj = MetaData()
    return Table(
        "participants",
        metadata_obj,
        Column("participant_id", String, primary_key=True),
        Column("gender", String),
        Column("is_onboarded", Boolean),
    )


def make_sample_data(n=100):
    """Create mock participant data that works with our sample_table"""
    rs = RandomState(MT19937())
    rs.seed(42)
    return [
        MockRow(
            participant_id=f"p{i}",
            gender=rs.choice(["M", "F"]),
            is_onboarded=bool(rs.choice([True, False], p=[0.5, 0.5])),
        )
        for i in range(n)
    ]


async def test_create_experiment_impl_for_preassigned(
    xngin_session: AsyncSession,
    testing_datasource,
    sample_table,
    use_deterministic_random,
):
    """Test implementation of creating a preassigned experiment."""
    participants = make_sample_data(n=100)
    request = make_create_preassigned_experiment_request(with_ids=True)
    # Add a partial mock PowerResponse just to verify storage
    request.power_analyses = PowerResponse(
        analyses=[
            MetricPowerAnalysis(metric_spec=DesignSpecMetric(field_name="is_onboarded", metric_type=MetricType.BINARY))
        ]
    )

    # Test!
    response = await create_preassigned_experiment_impl(
        request=request.model_copy(deep=True),  # we'll use the original request for assertions
        datasource_id=testing_datasource.ds.id,
        organization_id=testing_datasource.ds.organization_id,
        participant_unique_id_field="participant_id",
        dwh_sa_table=sample_table,
        dwh_participants=participants,
        random_state=42,
        xngin_session=xngin_session,
        stratify_on_metrics=True,
        validated_webhooks=[],
    )
    # Verify response
    assert response.datasource_id == testing_datasource.ds.id
    assert response.state == ExperimentState.ASSIGNED
    # Verify design_spec
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    assert response.design_spec.arms[1].arm_id is not None
    assert response.design_spec.experiment_name == request.design_spec.experiment_name
    assert response.design_spec.description == request.design_spec.description
    assert response.design_spec.start_date == request.design_spec.start_date
    assert response.design_spec.end_date == request.design_spec.end_date
    # although we stratify on target metrics as well in this test, note that the
    # original strata are not augmented with the metric names.
    assert response.design_spec.experiment_type == ExperimentsType.FREQ_PREASSIGNED
    assert isinstance(response.design_spec, PreassignedFrequentistExperimentSpec)
    assert response.design_spec.strata == [Stratum(field_name="gender")]
    assert response.power_analyses is not None
    assert response.power_analyses == request.power_analyses
    # Verify assign_summary
    assert response.assign_summary is not None
    assert response.assign_summary.sample_size == len(participants)
    assert response.assign_summary.balance_check is not None
    assert response.assign_summary.balance_check.balance_ok is True

    # Verify database state using the ids in the returned DesignSpec.
    experiment = await xngin_session.get(tables.Experiment, response.design_spec.experiment_id)
    assert experiment is not None
    assert experiment.experiment_type == ExperimentsType.FREQ_PREASSIGNED
    assert experiment.participant_type == request.design_spec.participant_type
    assert experiment.name == request.design_spec.experiment_name
    assert experiment.description == request.design_spec.description
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == testing_datasource.ds.id
    # This comparison is dependent on whether the db can store tz or not (sqlite does not).
    assert_dates_equal(experiment.start_date, request.design_spec.start_date)
    assert_dates_equal(experiment.end_date, request.design_spec.end_date)

    # Verify stats parameters were stored correctly
    assert isinstance(request.design_spec, PreassignedFrequentistExperimentSpec)
    assert experiment.power == request.design_spec.power
    assert experiment.alpha == request.design_spec.alpha
    assert experiment.fstat_thresh == request.design_spec.fstat_thresh
    # Verify design_spec was stored correctly
    converter = ExperimentStorageConverter(experiment)
    assert converter.get_design_spec() == response.design_spec
    assert converter.get_power_response() == response.power_analyses

    # Verify assignments were created
    assignments = (
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment.id)
        )
    ).all()
    assert len(assignments) == len(participants)
    # Verify all participant IDs in the db are the participants in the request
    assignment_participant_ids = {a.participant_id for a in assignments}
    assert assignment_participant_ids == {p.participant_id for p in participants}
    assert len(assignment_participant_ids) == len(participants)

    # Verify arms were created in database
    arms = (await xngin_session.scalars(select(tables.Arm).where(tables.Arm.experiment_id == experiment.id))).all()
    assert len(arms) == 2
    arm_ids = {arm.id for arm in arms}
    expected_arm_ids = {response_arm.arm_id for response_arm in response.design_spec.arms}
    assert arm_ids == expected_arm_ids

    # Check one assignment to see if it looks roughly right
    sample_assignment = assignments[0]
    assert sample_assignment.participant_type == "test_participant_type"
    assert sample_assignment.experiment_id == experiment.id
    assert sample_assignment.arm_id in (arm.arm_id for arm in response.design_spec.arms)
    # Verify strata information
    assert len(sample_assignment.strata) == 2  # our metric by default and the original strata
    assert sample_assignment.strata[0]["field_name"] == "gender"
    assert sample_assignment.strata[1]["field_name"] == "is_onboarded"

    # Check for approximate balance in arm assignments
    arm1_id = response.design_spec.arms[0].arm_id
    arm2_id = response.design_spec.arms[1].arm_id
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 1


async def test_create_preassigned_experiment_impl_raises_on_duplicate_ids(
    xngin_session: AsyncSession,
    testing_datasource,
    sample_table,
    use_deterministic_random,
):
    """Test that create_preassigned_experiment_impl raises LateValidationError for duplicate participant IDs."""
    request = make_create_preassigned_experiment_request(with_ids=True)

    # Create mock participants with a duplicate ID
    participants_with_duplicate = [
        MockRow(participant_id="id_1", gender="M", is_onboarded=True),
        MockRow(participant_id="id_2", gender="F", is_onboarded=False),
        MockRow(participant_id="id_1", gender="F", is_onboarded=True),  # Duplicate ID
    ]

    with pytest.raises(LateValidationError, match="Duplicate participant ID found after filtering:"):
        await create_preassigned_experiment_impl(
            request=request,
            datasource_id=testing_datasource.ds.id,
            organization_id=testing_datasource.ds.organization_id,
            participant_unique_id_field="participant_id",
            dwh_sa_table=sample_table,
            dwh_participants=participants_with_duplicate,
            random_state=42,
            xngin_session=xngin_session,
            stratify_on_metrics=False,
            validated_webhooks=[],
        )


async def test_create_experiment_impl_for_online(
    xngin_session, testing_datasource, sample_table, use_deterministic_random
):
    """Test implementation of creating an online experiment."""
    request = make_create_online_experiment_request()

    response = await create_dwh_experiment_impl(
        request=request.model_copy(deep=True),
        datasource=testing_datasource.ds,
        random_state=42,
        chosen_n=None,
        xngin_session=xngin_session,
        stratify_on_metrics=True,
        validated_webhooks=[],
    )
    # Verify response
    assert response.datasource_id == testing_datasource.ds.id
    assert response.state == ExperimentState.ASSIGNED

    # Verify design_spec
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    assert response.design_spec.arms[1].arm_id is not None
    assert response.design_spec.experiment_name == request.design_spec.experiment_name
    assert response.design_spec.description == request.design_spec.description
    assert response.design_spec.start_date == request.design_spec.start_date
    assert response.design_spec.end_date == request.design_spec.end_date
    assert isinstance(response.design_spec, OnlineFrequentistExperimentSpec)
    assert response.design_spec.strata == [Stratum(field_name="gender")]
    # Online experiments don't have power analyses by default
    assert response.power_analyses is None

    # Verify assign_summary for online experiment
    assert response.assign_summary is not None
    assert response.assign_summary.sample_size == 0
    assert response.assign_summary.balance_check is None
    assert response.assign_summary.arm_sizes is not None
    assert all(arm_size.size == 0 for arm_size in response.assign_summary.arm_sizes)

    # Verify database state
    experiment = await xngin_session.get(tables.Experiment, response.design_spec.experiment_id)
    assert experiment.experiment_type == ExperimentsType.FREQ_ONLINE
    assert experiment.participant_type == request.design_spec.participant_type
    assert experiment.name == request.design_spec.experiment_name
    assert experiment.description == request.design_spec.description
    # Online experiments still go through a review step before being committed
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == testing_datasource.ds.id
    assert_dates_equal(experiment.start_date, request.design_spec.start_date)
    assert_dates_equal(experiment.end_date, request.design_spec.end_date)
    # Verify stats parameters were stored correctly
    assert isinstance(request.design_spec, OnlineFrequentistExperimentSpec)
    assert experiment.power == request.design_spec.power
    assert experiment.alpha == request.design_spec.alpha
    assert experiment.fstat_thresh == request.design_spec.fstat_thresh
    # Verify design_spec was stored correctly
    converter = ExperimentStorageConverter(experiment)
    assert converter.get_design_spec() == response.design_spec
    # Verify no power_analyses for online experiments
    assert experiment.power_analyses is None

    # Verify arms were created in database
    arms = (await xngin_session.scalars(select(tables.Arm).where(tables.Arm.experiment_id == experiment.id))).all()
    assert len(arms) == 2
    arm_ids = {arm.id for arm in arms}
    expected_arm_ids = {arm.arm_id for arm in response.design_spec.arms}
    assert arm_ids == expected_arm_ids

    # Verify that no assignments were created for online experiment
    assignments = (
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment.id)
        )
    ).all()
    assert len(assignments) == 0


async def test_create_experiment_impl_for_mab_online(xngin_session, testing_datasource):
    """Test implementation of creating an online experiment."""
    request = make_create_online_bandit_experiment_request(with_ids=True)

    response = await create_bandit_online_experiment_impl(
        request=request.model_copy(deep=True),
        chosen_n=None,
        xngin_session=xngin_session,
        organization_id=testing_datasource.org.id,
        datasource_id=testing_datasource.ds.id,
        validated_webhooks=[],
    )
    # Verify response
    assert response.datasource_id == testing_datasource.ds.id
    assert response.state == ExperimentState.ASSIGNED

    # Verify design_spec
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    assert response.design_spec.arms[1].arm_id is not None
    assert response.design_spec.experiment_name == request.design_spec.experiment_name
    assert response.design_spec.description == request.design_spec.description
    assert response.design_spec.start_date == request.design_spec.start_date
    assert response.design_spec.end_date == request.design_spec.end_date
    assert isinstance(response.design_spec, MABExperimentSpec)
    assert response.assign_summary is None

    # Verify database state
    experiment = await xngin_session.get(tables.Experiment, response.design_spec.experiment_id)
    assert experiment.experiment_type == ExperimentsType.MAB_ONLINE
    assert experiment.participant_type == request.design_spec.participant_type
    assert experiment.name == request.design_spec.experiment_name
    assert experiment.description == request.design_spec.description
    # Online experiments still go through a review step before being committed
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == testing_datasource.ds.id
    assert_dates_equal(experiment.start_date, request.design_spec.start_date)
    assert_dates_equal(experiment.end_date, request.design_spec.end_date)

    # Verify design_spec was stored correctly
    converter = ExperimentStorageConverter(experiment)
    converted_design_spec = converter.get_design_spec()
    assert converted_design_spec == response.design_spec
    assert isinstance(converted_design_spec, MABExperimentSpec)
    for arms in converted_design_spec.arms:
        if response.design_spec.prior_type == PriorTypes.NORMAL:
            assert arms.mu is not None
            assert arms.covariance is not None
        elif response.design_spec.prior_type == PriorTypes.BETA:
            assert arms.alpha is not None
            assert arms.beta is not None

    # Verify arms were created in database
    arms = (await xngin_session.scalars(select(tables.Arm).where(tables.Arm.experiment_id == experiment.id))).all()
    assert len(arms) == 2
    arm_ids = {arm.id for arm in arms}
    expected_arm_ids = {arm.arm_id for arm in response.design_spec.arms}
    assert arm_ids == expected_arm_ids

    # Verify that no assignments were created for online experiment
    assignments = (
        await xngin_session.scalars(select(tables.Draw).where(tables.Draw.experiment_id == experiment.id))
    ).all()
    assert len(assignments) == 0


async def test_create_experiment_impl_for_cmab_online(xngin_session, testing_datasource):
    """Test implementation of creating an online experiment."""
    request = make_create_online_bandit_experiment_request(experiment_type=ExperimentsType.CMAB_ONLINE, with_ids=True)

    response = await create_bandit_online_experiment_impl(
        request=request.model_copy(deep=True),
        chosen_n=None,
        xngin_session=xngin_session,
        organization_id=testing_datasource.org.id,
        datasource_id=testing_datasource.ds.id,
        validated_webhooks=[],
    )
    # Verify response
    assert response.datasource_id == testing_datasource.ds.id
    assert response.state == ExperimentState.ASSIGNED

    # Verify design_spec
    assert isinstance(response.design_spec, CMABExperimentSpec)
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    assert response.design_spec.arms[1].arm_id is not None
    assert response.design_spec.contexts is not None
    assert len(response.design_spec.contexts) == 2
    assert response.design_spec.contexts[0].context_id is not None
    assert response.design_spec.contexts[1].context_id is not None
    assert response.design_spec.experiment_name == request.design_spec.experiment_name
    assert response.design_spec.description == request.design_spec.description
    assert response.design_spec.start_date == request.design_spec.start_date
    assert response.design_spec.end_date == request.design_spec.end_date
    assert response.assign_summary is None

    # Verify database state
    experiment = await xngin_session.get(tables.Experiment, response.design_spec.experiment_id)
    assert experiment.experiment_type == ExperimentsType.CMAB_ONLINE
    assert experiment.participant_type == request.design_spec.participant_type
    assert experiment.name == request.design_spec.experiment_name
    assert experiment.description == request.design_spec.description
    # Online experiments still go through a review step before being committed
    assert experiment.state == ExperimentState.ASSIGNED
    assert experiment.datasource_id == testing_datasource.ds.id
    assert_dates_equal(experiment.start_date, request.design_spec.start_date)
    assert_dates_equal(experiment.end_date, request.design_spec.end_date)

    # Verify design_spec was stored correctly
    converter = ExperimentStorageConverter(experiment)
    converted_design_spec = converter.get_design_spec()
    assert converted_design_spec == response.design_spec
    assert isinstance(converted_design_spec, CMABExperimentSpec)
    assert converted_design_spec.prior_type == PriorTypes.NORMAL
    for arms in converted_design_spec.arms:
        assert arms.mu is not None and len(arms.mu) == 2
        assert arms.covariance is not None and np.array(arms.covariance).size == 4

    assert converted_design_spec.contexts is not None
    for context in converted_design_spec.contexts:
        assert context.context_id is not None

    # Verify arms were created in database
    arms = (await xngin_session.scalars(select(tables.Arm).where(tables.Arm.experiment_id == experiment.id))).all()
    assert len(arms) == 2
    arm_ids = {arm.id for arm in arms}
    expected_arm_ids = {arm.arm_id for arm in response.design_spec.arms}
    assert arm_ids == expected_arm_ids

    # Verify contexts were created in database
    contexts = (
        await xngin_session.scalars(select(tables.Context).where(tables.Context.experiment_id == experiment.id))
    ).all()
    assert len(contexts) == 2
    context_ids = {context.id for context in contexts}
    expected_context_ids = {context.context_id for context in response.design_spec.contexts}
    assert context_ids == expected_context_ids

    # Verify that no assignments were created for online experiment
    assignments = (
        await xngin_session.scalars(select(tables.Draw).where(tables.Draw.experiment_id == experiment.id))
    ).all()
    assert len(assignments) == 0


async def test_create_experiment_impl_overwrites_uuids(
    xngin_session, testing_datasource, sample_table, use_deterministic_random
):
    """
    Test that the function overwrites requests with preset UUIDs
    (which would otherwise be caught in the route handler).
    """
    participants = make_sample_data(n=100)
    request = make_create_preassigned_experiment_request(with_ids=True)
    original_experiment_id = request.design_spec.experiment_id
    original_arm_ids = [arm.arm_id for arm in request.design_spec.arms]

    response = await create_dwh_experiment_impl(
        request=request,
        datasource=testing_datasource.ds,
        random_state=42,
        xngin_session=xngin_session,
        chosen_n=len(participants),
        stratify_on_metrics=True,
        validated_webhooks=[],
    )

    # Verify that new UUIDs were generated
    assert response.design_spec.experiment_id != original_experiment_id
    new_arm_ids = [arm.arm_id for arm in response.design_spec.arms]
    assert set(new_arm_ids) != set(original_arm_ids)

    # Verify database state
    experiment = (
        await xngin_session.scalars(
            select(tables.Experiment).where(tables.Experiment.id == response.design_spec.experiment_id)
        )
    ).one()
    assert experiment.state == ExperimentState.ASSIGNED
    # Verify assignments were created with the new UUIDs
    assignments = (
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment.id)
        )
    ).all()
    # Verify all assignments use the new arm IDs
    assignment_arm_ids = {a.arm_id for a in assignments}
    assert assignment_arm_ids == set(new_arm_ids)


async def test_create_experiment_impl_no_metric_stratification(
    xngin_session, testing_datasource, sample_table, use_deterministic_random
):
    """Test implementation of creating an experiment without stratifying on metrics."""
    participants = make_sample_data(n=100)
    request = make_create_preassigned_experiment_request()

    # Test with stratify_on_metrics=False
    response = await create_dwh_experiment_impl(
        request=request.model_copy(deep=True),
        datasource=testing_datasource.ds,
        random_state=42,
        xngin_session=xngin_session,
        chosen_n=len(participants),
        stratify_on_metrics=False,
        validated_webhooks=[],
    )

    # Verify basic response
    assert response.datasource_id == testing_datasource.ds.id
    assert response.state == ExperimentState.ASSIGNED
    assert response.design_spec.experiment_id is not None
    assert response.design_spec.arms[0].arm_id is not None
    # Same as in the stratify_on_metrics=True test.
    # Only the output assignments will also store a snapshot of the metric values as strata.
    assert isinstance(response.design_spec, PreassignedFrequentistExperimentSpec)
    assert response.design_spec.strata == [Stratum(field_name="gender")]

    # Verify database state
    experiment = (
        await xngin_session.scalars(
            select(tables.Experiment).where(tables.Experiment.id == response.design_spec.experiment_id)
        )
    ).one()
    # Verify assignments were created
    assignments = (
        await xngin_session.scalars(
            select(tables.ArmAssignment).where(tables.ArmAssignment.experiment_id == experiment.id)
        )
    ).all()
    assert len(assignments) == len(participants)
    # Check strata information only has gender, not is_onboarded
    sample_assignment = assignments[0]
    assert len(sample_assignment.strata) == 1
    assert sample_assignment.strata[0]["field_name"] == "gender"
    assert not any(s["field_name"] == "is_onboarded" for s in sample_assignment.strata)

    # Check for approximate balance in arm assignments
    arm1_id = response.design_spec.arms[0].arm_id
    arm2_id = response.design_spec.arms[1].arm_id
    num_control = sum(1 for a in assignments if a.arm_id == arm1_id)
    num_treat = sum(1 for a in assignments if a.arm_id == arm2_id)
    assert abs(num_control - num_treat) <= 1


@pytest.mark.parametrize(
    "method_under_test,initial_state,expected_state,expected_status,expected_detail",
    [
        # Success case
        (
            commit_experiment_impl,
            ExperimentState.ASSIGNED,
            ExperimentState.COMMITTED,
            204,
            None,
        ),
        # No-op
        (
            commit_experiment_impl,
            ExperimentState.COMMITTED,
            ExperimentState.COMMITTED,
            304,
            None,
        ),
        # Failure cases
        (
            commit_experiment_impl,
            ExperimentState.DESIGNING,
            ExperimentState.DESIGNING,
            400,
            "Invalid state: designing",
        ),
        (
            commit_experiment_impl,
            ExperimentState.ABORTED,
            ExperimentState.ABORTED,
            400,
            "Invalid state: aborted",
        ),
        # Success cases
        (
            abandon_experiment_impl,
            ExperimentState.DESIGNING,
            ExperimentState.ABANDONED,
            204,
            None,
        ),
        (
            abandon_experiment_impl,
            ExperimentState.ASSIGNED,
            ExperimentState.ABANDONED,
            204,
            None,
        ),
        # No-op
        (
            abandon_experiment_impl,
            ExperimentState.ABANDONED,
            ExperimentState.ABANDONED,
            304,
            None,
        ),
        # Failure case
        (
            abandon_experiment_impl,
            ExperimentState.COMMITTED,
            ExperimentState.COMMITTED,
            400,
            "Invalid state: committed",
        ),
    ],
)
async def test_state_setting_experiment_impl(
    xngin_session,
    testing_datasource,
    method_under_test,
    initial_state,
    expected_state,
    expected_status,
    expected_detail,
):
    # Initialize our state with an existing experiment who's state we want to modify.
    experiment = await insert_experiment_and_arms(xngin_session, testing_datasource.ds, state=initial_state)

    try:
        response = await method_under_test(xngin_session, experiment)
    except HTTPException as e:
        assert e.status_code == expected_status
        assert e.detail == expected_detail
    else:
        assert response.status_code == expected_status
        assert experiment.state == expected_state


async def test_list_experiments_impl(
    xngin_session,
    testing_datasource,
    testing_datasource_with_user,
):
    """Test that we only get experiments in a valid state for the specified datasource."""
    experiment1_data = make_insertable_experiment(testing_datasource.ds, ExperimentState.ASSIGNED)
    experiment2_data = make_insertable_experiment(testing_datasource.ds, ExperimentState.COMMITTED)
    experiment3_data = make_insertable_experiment(testing_datasource.ds, ExperimentState.DESIGNING)
    experiment4_data = make_insertable_experiment(testing_datasource.ds, ExperimentState.ABORTED)
    # One more experiment associated with a *different* datasource.
    experiment5_data = make_insertable_experiment(testing_datasource_with_user.ds, ExperimentState.ASSIGNED)
    # Set the created_at time to test ordering
    experiment1_data[0].created_at = datetime.now(UTC) - timedelta(days=1)
    experiment2_data[0].created_at = datetime.now(UTC)
    experiment3_data[0].created_at = datetime.now(UTC) + timedelta(days=1)
    experiment_data = [
        experiment1_data,
        experiment2_data,
        experiment3_data,
        experiment4_data,
        experiment5_data,
    ]

    xngin_session.add_all([data[0] for data in experiment_data])
    await xngin_session.commit()

    experiments = await list_organization_or_datasource_experiments_impl(
        xngin_session=xngin_session, datasource_id=testing_datasource.ds.id
    )
    # experiment5 excluded due to datasource mismatch
    assert len(experiments.items) == 3

    # Verify that the experiments are in the correct order
    actual1_config = experiments.items[2]  # experiment1 is last as it's oldest
    actual2_config = experiments.items[1]
    actual3_config = experiments.items[0]
    assert actual1_config.state == ExperimentState.ASSIGNED
    diff = DeepDiff(actual1_config.design_spec, experiment1_data[1])
    assert not diff, f"Objects differ:\n{diff.pretty()}"
    assert actual2_config.state == ExperimentState.COMMITTED
    diff = DeepDiff(actual2_config.design_spec, experiment2_data[1])
    assert not diff, f"Objects differ:\n{diff.pretty()}"
    assert actual3_config.state == ExperimentState.DESIGNING
    diff = DeepDiff(actual3_config.design_spec, experiment3_data[1])
    assert not diff, f"Objects differ:\n{diff.pretty()}"


async def test_list_experiments_impl_alt_scenarios(
    xngin_session,
    testing_datasource,
):
    with pytest.raises(ValueError, match="Either datasource_id or organization_id must be provided"):
        await list_organization_or_datasource_experiments_impl(xngin_session=xngin_session)

    experiment1_data = make_insertable_experiment(testing_datasource.ds, ExperimentState.ASSIGNED)
    xngin_session.add(experiment1_data[0])
    org_list = await list_organization_or_datasource_experiments_impl(
        xngin_session=xngin_session, organization_id=testing_datasource.org.id
    )
    ds_list = await list_organization_or_datasource_experiments_impl(
        xngin_session=xngin_session,
        datasource_id=testing_datasource.ds.id,
    )
    both_list = await list_organization_or_datasource_experiments_impl(
        xngin_session=xngin_session,
        datasource_id=testing_datasource.ds.id,
        organization_id=testing_datasource.org.id,
    )
    assert len(org_list.items) == 1
    assert len(ds_list.items) == 1
    assert len(both_list.items) == 1
    assert org_list == ds_list
    assert ds_list == both_list

    bad_list = await list_organization_or_datasource_experiments_impl(
        xngin_session=xngin_session,
        datasource_id="bad_id",
        organization_id=testing_datasource.org.id,
    )
    assert bad_list.items == []


async def test_get_experiment_assignments_impl(xngin_session, testing_datasource):
    # First insert an experiment with assignments
    experiment = await insert_experiment_and_arms(xngin_session, testing_datasource.ds)
    await xngin_session.commit()

    experiment_id = experiment.id
    arm1_id = experiment.arms[0].id
    arm2_id = experiment.arms[1].id
    arm_assignments = [
        tables.ArmAssignment(
            experiment_id=experiment_id,
            participant_type="test_participant_type",
            participant_id="p1",
            arm_id=arm1_id,
            strata=[{"field_name": "gender", "strata_value": "F"}],
        ),
        tables.ArmAssignment(
            experiment_id=experiment_id,
            participant_type="test_participant_type",
            participant_id="p2",
            arm_id=arm2_id,
            strata=[{"field_name": "gender", "strata_value": "M"}],
        ),
    ]
    xngin_session.add_all(arm_assignments)
    await xngin_session.commit()
    await xngin_session.refresh(experiment, ["arms", "arm_assignments"])

    data = get_experiment_assignments_impl(experiment)

    # Check the response structure
    assert data.experiment_id == experiment.id
    assert data.sample_size == (await get_assign_summary(xngin_session, experiment.id)).sample_size
    assert data.balance_check == ExperimentStorageConverter(experiment).get_balance_check()

    # Check assignments
    assignments = data.assignments
    assert len(assignments) == 2

    # Verify first assignment
    assert assignments[0].participant_id == "p1"
    assert str(assignments[0].arm_id) == arm1_id
    assert assignments[0].arm_name == "control"
    assert assignments[0].strata is not None and len(assignments[0].strata) == 1
    assert assignments[0].strata[0].field_name == "gender"
    assert assignments[0].strata[0].strata_value == "F"
    assert assignments[0].created_at is not None

    # Verify second assignment
    assert assignments[1].participant_id == "p2"
    assert str(assignments[1].arm_id) == arm2_id
    assert assignments[1].arm_name == "treatment"
    assert assignments[1].strata is not None and len(assignments[1].strata) == 1
    assert assignments[1].strata[0].field_name == "gender"
    assert assignments[1].strata[0].strata_value == "M"
    assert assignments[1].created_at is not None


async def test_get_experiment_mab_assignments_impl(xngin_session, testing_datasource):
    # First insert an experiment with assignments
    experiment = await insert_experiment_and_arms(
        xngin_session, testing_datasource.ds, experiment_type=ExperimentsType.MAB_ONLINE
    )
    await xngin_session.commit()

    experiment_id = experiment.id
    arm1_id = experiment.arms[0].id
    arm2_id = experiment.arms[1].id
    arm_assignments = [
        tables.Draw(
            experiment_id=experiment_id,
            participant_type="test_participant_type",
            participant_id="p1",
            arm_id=arm1_id,
            current_mu=experiment.arms[0].mu,
            current_covariance=experiment.arms[0].covariance,
        ),
        tables.Draw(
            experiment_id=experiment_id,
            participant_type="test_participant_type",
            participant_id="p2",
            arm_id=arm2_id,
            current_mu=experiment.arms[1].mu,
            current_covariance=experiment.arms[1].covariance,
        ),
    ]
    xngin_session.add_all(arm_assignments)
    await xngin_session.commit()
    await xngin_session.refresh(experiment, ["arms", "draws"])

    data = get_experiment_assignments_impl(experiment)

    # Check the response structure
    assert data.experiment_id == experiment.id

    # Check assignments
    assignments = data.assignments
    assert len(assignments) == 2

    # Verify first assignment
    assert assignments[0].participant_id == "p1"
    assert str(assignments[0].arm_id) == arm1_id
    assert assignments[0].arm_name == "string"

    # Verify second assignment
    assert assignments[1].participant_id == "p2"
    assert str(assignments[1].arm_id) == arm2_id
    assert assignments[1].arm_name == "string"

    for assignment in assignments:
        assert assignment.outcome is None
        assert assignment.context_values is None
        assert assignment.observed_at is None
        assert assignment.created_at is not None


async def make_experiment_with_assignments(xngin_session, datasource: tables.Datasource):
    """Helper test function that commits a new preassigned experiment with assignments."""
    experiment = await insert_experiment_and_arms(xngin_session, datasource)
    arm1_id = experiment.arms[0].id
    arm2_id = experiment.arms[1].id
    assignments = [
        tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_type="test_participant_type",
            participant_id="p1",
            arm_id=arm1_id,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            strata=[
                {"field_name": "gender", "strata_value": "F"},
                {"field_name": "score", "strata_value": "1.1"},
            ],
        ),
        tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_type="test_participant_type",
            participant_id="p2",
            arm_id=arm2_id,
            created_at=datetime(2025, 1, 2, tzinfo=UTC),
            strata=[
                {"field_name": "gender", "strata_value": "M"},
                {"field_name": "score", "strata_value": "esc,aped"},
            ],
        ),
    ]
    xngin_session.add_all(assignments)
    await xngin_session.commit()
    return experiment


async def test_experiment_assignments_to_csv_generator(xngin_session, testing_datasource):
    experiment = await make_experiment_with_assignments(xngin_session, testing_datasource.ds)
    await xngin_session.refresh(experiment, ["arms", "arm_assignments"])

    arm_name_to_id = {a.name: a.id for a in experiment.arms}
    batches = list(experiment_assignments_to_csv_generator(experiment)())
    assert len(batches) == 1
    rows = batches[0].splitlines(keepends=True)
    assert rows[0] == "participant_id,arm_id,arm_name,created_at,gender,score\r\n"
    assert rows[1] == f"p1,{arm_name_to_id['control']},control,2025-01-01 00:00:00+00:00,F,1.1\r\n"
    assert rows[2] == f'p2,{arm_name_to_id["treatment"]},treatment,2025-01-02 00:00:00+00:00,M,"esc,aped"\r\n'


async def test_get_existing_assignment_for_participant(xngin_session, testing_datasource):
    experiment = await make_experiment_with_assignments(xngin_session, testing_datasource.ds)
    await xngin_session.refresh(experiment, ["arm_assignments"])
    expected_assignment = experiment.arm_assignments[0]

    assignment = await get_existing_assignment_for_participant(
        xngin_session,
        experiment.id,
        expected_assignment.participant_id,
        experiment.experiment_type,
    )
    assert assignment is not None
    assert assignment.participant_id == expected_assignment.participant_id
    assert str(assignment.arm_id) == expected_assignment.arm_id

    assignment = await get_existing_assignment_for_participant(
        xngin_session, experiment.id, "new_id", experiment.experiment_type
    )
    assert assignment is None


async def test_create_assignment_for_participant_errors(xngin_session, testing_datasource):
    # Test assignment while in an experiment state not valid for assignments.
    # Preassigned will short circuit before the invalid state check so will NOT raise.
    experiment, _ = make_insertable_experiment(
        testing_datasource.ds,
        ExperimentState.ASSIGNED,
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
    )
    experiment.arms = []
    response = await create_assignment_for_participant(xngin_session, experiment, "p1", None)
    assert response is None

    # But an online experiment in this invalid state will raise.
    experiment, _ = make_insertable_experiment(
        testing_datasource.ds,
        ExperimentState.ASSIGNED,
        experiment_type=ExperimentsType.FREQ_ONLINE,
    )
    with pytest.raises(ExperimentsAssignmentError, match="Invalid experiment state: assigned"):
        await create_assignment_for_participant(xngin_session, experiment, "p1", None)

    # Test that an online experiment with no arms will raise.
    experiment, _ = make_insertable_experiment(
        testing_datasource.ds,
        ExperimentState.COMMITTED,
        experiment_type=ExperimentsType.FREQ_ONLINE,
    )
    experiment.arms = []
    with pytest.raises(ExperimentsAssignmentError, match="Experiment has no arms"):
        await create_assignment_for_participant(xngin_session, experiment, "p1", None)


async def test_create_assignment_for_participant(xngin_session, testing_datasource):
    preassigned_experiment = await insert_experiment_and_arms(xngin_session, testing_datasource.ds)
    # Assert that we won't create new assignments for preassigned experiments
    expect_none = await create_assignment_for_participant(xngin_session, preassigned_experiment, "new_id", None)
    assert expect_none is None

    # Test create assignment for online frequentist and bandit experiments
    freq_online_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.FREQ_ONLINE,
    )
    assignment_freq_online = await create_assignment_for_participant(xngin_session, freq_online_experiment, "new_id")

    mab_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=ExperimentsType.MAB_ONLINE,
    )
    mab_assignment = await create_assignment_for_participant(xngin_session, mab_experiment, "new_id")

    # For frequentist experiments
    # Assert that we do create new assignments for online experiments
    assert assignment_freq_online is not None
    assert assignment_freq_online.participant_id == "new_id"
    freq_online_arm_map = {arm.id: arm.name for arm in freq_online_experiment.arms}
    assert assignment_freq_online.arm_name == freq_online_arm_map[str(assignment_freq_online.arm_id)]
    assert not assignment_freq_online.strata

    # But that if we try to create an assignment for a participant that already has one, it triggers an error.
    with pytest.raises(ExperimentsAssignmentError, match="Failed to assign participant"):
        await create_assignment_for_participant(xngin_session, freq_online_experiment, "new_id")

    # For MAB experiments
    # Assert that we do create new assignments for online MAB experiments
    assert mab_assignment is not None
    assert mab_assignment.participant_id == "new_id"
    mab_arms = await mab_experiment.awaitable_attrs.arms
    mab_arm_map = {arm.id: arm.name for arm in mab_arms}
    assert mab_assignment.arm_name == mab_arm_map[str(mab_assignment.arm_id)]
    assert not mab_assignment.context_values
    assert mab_assignment.created_at is not None

    # But that if we try to create an assignment for a participant that already has one, it triggers an error.
    with pytest.raises(ExperimentsAssignmentError, match="Failed to assign participant"):
        await create_assignment_for_participant(xngin_session, mab_experiment, "new_id")


@pytest.mark.parametrize(
    "experiment_type,stopped_reason",
    [
        (ExperimentsType.FREQ_PREASSIGNED, StopAssignmentReason.PREASSIGNED),
        (ExperimentsType.FREQ_ONLINE, StopAssignmentReason.END_DATE),
        (ExperimentsType.MAB_ONLINE, StopAssignmentReason.END_DATE),
        (ExperimentsType.CMAB_ONLINE, StopAssignmentReason.END_DATE),
    ],
)
async def test_create_assignment_for_participant_stopped_reason(
    xngin_session, testing_datasource, experiment_type, stopped_reason
):
    experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=experiment_type,
        end_date=datetime.now(UTC) - timedelta(days=1),
    )

    # Assert that we don't create assignments for experiments in the past,
    # but for preassigned experiments we don't set a stopped_reason.
    assignment = await create_assignment_for_participant(
        xngin_session, experiment, "new_id", [1.0, 1.0] if experiment_type == ExperimentsType.CMAB_ONLINE else None
    )
    assert assignment is None
    assert experiment.stopped_assignments_reason == stopped_reason
    if stopped_reason is not None:
        assert datetime.now(UTC) - experiment.stopped_assignments_at < timedelta(seconds=1)
    else:
        assert experiment.stopped_assignments_at is None


@pytest.mark.parametrize(
    "experiment_type,prior_type,reward_type",
    [
        (ExperimentsType.MAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.NORMAL),
        (ExperimentsType.MAB_ONLINE, PriorTypes.BETA, LikelihoodTypes.BERNOULLI),
        (ExperimentsType.MAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI),
        (ExperimentsType.CMAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.NORMAL),
        (ExperimentsType.CMAB_ONLINE, PriorTypes.NORMAL, LikelihoodTypes.BERNOULLI),
    ],
)
async def test_update_bandit_arm_with_outcome(
    xngin_session, testing_datasource, experiment_type, prior_type, reward_type
):
    bandit_experiment = await insert_experiment_and_arms(
        xngin_session,
        testing_datasource.ds,
        experiment_type=experiment_type,
        prior_type=prior_type,
        reward_type=reward_type,
    )
    await create_assignment_for_participant(
        xngin_session,
        bandit_experiment,
        "test_id",
        [1.0, 1.0] if experiment_type == ExperimentsType.CMAB_ONLINE else None,
    )

    updated_arm = await update_bandit_arm_with_outcome_impl(
        xngin_session=xngin_session, experiment=bandit_experiment, participant_id="test_id", outcome=1.0
    )

    # Refresh experiment; retrieve draws
    await xngin_session.refresh(bandit_experiment)
    draws = await updated_arm.awaitable_attrs.draws
    draw = draws[0]

    # Assert that the draw was updated correctly
    assert len(draws) == 1
    assert draw.outcome == 1.0
    assert draw.observed_at is not None
    await bandit_experiment.awaitable_attrs.arms
    await bandit_experiment.awaitable_attrs.contexts
    bandit_arm_map = {arm.id: arm for arm in bandit_experiment.arms}
    assert draw.current_mu == bandit_arm_map[updated_arm.id].mu
    assert draw.current_covariance == bandit_arm_map[updated_arm.id].covariance
    assert draw.current_alpha == bandit_arm_map[updated_arm.id].alpha
    assert draw.current_beta == bandit_arm_map[updated_arm.id].beta

    if experiment_type == ExperimentsType.CMAB_ONLINE:
        assert draw.context_vals == [1.0, 1.0]

    # Assert that we can't update the arm with an outcome for a participant that doesn't exist
    with pytest.raises(
        ExperimentsAssignmentError,
        match="Participant {participant_id} does not have an assignment for which to record an outcome.".format(
            participant_id="some_other_id"
        ),
    ):
        await update_bandit_arm_with_outcome_impl(xngin_session, bandit_experiment, "some_other_id", 1.0)

    # Assert that we can't update the arm with an outcome for a participant that already has an outcome
    with pytest.raises(
        ExperimentsAssignmentError,
        match="Participant {participant_id} already has an outcome recorded.".format(participant_id="test_id"),
    ):
        await update_bandit_arm_with_outcome_impl(xngin_session, bandit_experiment, "test_id", 1.0)


def test_experiment_sql():
    pg_sql = str(CreateTable(cast(Table, tables.ArmAssignment.__table__)).compile(dialect=postgresql.dialect()))
    assert "arm_id VARCHAR(36) NOT NULL," in pg_sql
    assert "strata JSONB NOT NULL," in pg_sql
