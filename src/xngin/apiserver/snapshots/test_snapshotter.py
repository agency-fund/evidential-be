from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from xngin.apiserver.routers.common_api_types import (
    Arm,
    DesignSpec,
    DesignSpecMetricRequest,
    ExperimentsType,
    FreqExperimentAnalysisResponse,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import ExperimentState, StopAssignmentReason
from xngin.apiserver.snapshots.snapshotter import create_pending_snapshots, make_first_snapshot
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter


async def make_experiment(xngin_session, datasource: tables.Datasource, design_spec: DesignSpec) -> tables.Experiment:
    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource.id,
        organization_id=datasource.organization_id,
        experiment_type=design_spec.experiment_type,
        design_spec=design_spec,
        state=ExperimentState.COMMITTED,
        stopped_assignments_at=datetime.now(UTC),
        stopped_assignments_reason=StopAssignmentReason.PREASSIGNED,
    )
    experiment = experiment_converter.get_experiment()
    xngin_session.add(experiment)
    await xngin_session.commit()
    # Add assignments to the experiment so analysis doesn't error.
    arm_assignments = [
        tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_type="test_participant_type",
            participant_id="1",
            arm_id=experiment.arms[0].id,
            strata=[],
        ),
        tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_type="test_participant_type",
            participant_id="2",
            arm_id=experiment.arms[1].id,
            strata=[],
        ),
    ]
    xngin_session.add_all(arm_assignments)
    await xngin_session.commit()
    await xngin_session.refresh(experiment, ["arms", "arm_assignments"])
    return experiment


async def get_latest_snapshot_analysis(xngin_session, experiment_id):
    """Helper to fetch the latest snapshot analysis payload for an experiment."""
    snapshot = (
        await xngin_session.execute(
            select(tables.Snapshot)
            .where(tables.Snapshot.experiment_id == experiment_id)
            .order_by(tables.Snapshot.created_at.desc())
            .limit(1)
        )
    ).scalar_one()
    assert snapshot.status == "success"
    return FreqExperimentAnalysisResponse.model_validate(snapshot.data)


async def test_make_first_snapshot_of_freq_preassigned(xngin_session, testing_datasource_with_user):
    datasource = testing_datasource_with_user.ds

    # Create a preassigned frequentist experiment design spec
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        participant_type="test_participant_type",
        experiment_name="test experiment",
        description="test experiment",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC) + timedelta(days=1),
        arms=[
            Arm(arm_name="C", arm_description="C"),
            Arm(arm_name="T", arm_description="T"),
        ],
        metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=0.1)],
        strata=[],
        filters=[],
    )

    experiment = await make_experiment(xngin_session, datasource, design_spec)
    # Arms' intial position should reflect design spec ordering
    arm1 = experiment.arms[0]
    assert arm1.position == 1
    arm2 = experiment.arms[1]
    assert arm2.position == 2

    # Test 1: baseline arm is the position 1 arm.
    await create_pending_snapshots(5)
    snapshot_id = (
        await xngin_session.execute(
            select(tables.Snapshot.id)
            .where(tables.Snapshot.experiment_id == experiment.id)
            .where(tables.Snapshot.status == "pending")
        )
    ).scalar_one()
    await make_first_snapshot(experiment.id, snapshot_id)
    analysis = await get_latest_snapshot_analysis(xngin_session, experiment.id)
    # Verify analysis payload is in the order of experiment.arms above.
    assert isinstance(analysis, FreqExperimentAnalysisResponse)
    assert len(analysis.metric_analyses) == 1
    arm_analyses = analysis.metric_analyses[0].arm_analyses
    assert len(arm_analyses) == 2
    assert arm_analyses[0].arm_id == arm1.id
    assert arm_analyses[0].is_baseline
    assert arm_analyses[1].arm_id == arm2.id
    assert not arm_analyses[1].is_baseline

    # Test 2: swap the arm positions, verifying baseline switches
    arm1.position = 2
    arm2.position = 1
    await xngin_session.commit()
    await xngin_session.refresh(experiment, ["arms"])

    await create_pending_snapshots(0)  # Force new snapshot immediately
    snapshot_id = (
        await xngin_session.execute(
            select(tables.Snapshot.id)
            .where(tables.Snapshot.experiment_id == experiment.id)
            .where(tables.Snapshot.status == "pending")
        )
    ).scalar_one()
    await make_first_snapshot(experiment.id, snapshot_id)
    analysis = await get_latest_snapshot_analysis(xngin_session, experiment.id)

    assert isinstance(analysis, FreqExperimentAnalysisResponse)
    assert len(analysis.metric_analyses) == 1
    arm_analyses = analysis.metric_analyses[0].arm_analyses
    assert len(arm_analyses) == 2
    assert arm_analyses[0].arm_id == arm2.id
    assert arm_analyses[0].is_baseline
    assert arm_analyses[1].arm_id == arm1.id
    assert not arm_analyses[1].is_baseline

    # Test 3: wipe the positions, verifying baseline is the arm in the 0th index
    arm1.position = None
    arm2.position = None
    await xngin_session.commit()
    await xngin_session.refresh(experiment, ["arms"])
    assert experiment.arms[0].position is None
    assert experiment.arms[1].position is None

    await create_pending_snapshots(0)  # Force new snapshot immediately
    snapshot_id = (
        await xngin_session.execute(
            select(tables.Snapshot.id)
            .where(tables.Snapshot.experiment_id == experiment.id)
            .where(tables.Snapshot.status == "pending")
        )
    ).scalar_one()
    await make_first_snapshot(experiment.id, snapshot_id)
    analysis = await get_latest_snapshot_analysis(xngin_session, experiment.id)

    assert isinstance(analysis, FreqExperimentAnalysisResponse)
    assert len(analysis.metric_analyses) == 1
    arm_analyses = analysis.metric_analyses[0].arm_analyses
    # With no position info, the order of experiment.arms is not guaranteed,
    # but we expect baseline to be arm 0.
    baseline_arm = next(a for a in arm_analyses if a.is_baseline)
    non_baseline_arm = next(a for a in arm_analyses if not a.is_baseline)
    assert baseline_arm.arm_id == experiment.arms[0].id
    assert non_baseline_arm.arm_id == experiment.arms[1].id
