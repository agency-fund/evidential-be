import warnings
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from xngin.apiserver.routers.admin.admin_api_types import SnapshotStatus
from xngin.apiserver.routers.common_api_types import (
    Arm,
    BaseFrequentistDesignSpec,
    CreateExperimentRequest,
    DesignSpec,
    DesignSpecMetricRequest,
    ExperimentsType,
    FreqExperimentAnalysisResponse,
    PreassignedFrequentistExperimentSpec,
)
from xngin.apiserver.routers.common_enums import ExperimentState, StopAssignmentReason
from xngin.apiserver.routers.experiments.experiments_common import fetch_fields_or_raise
from xngin.apiserver.snapshots.snapshotter import (
    SNAPSHOT_TIMEOUT_SECS,
    create_pending_snapshots,
    make_first_snapshot,
    process_pending_snapshots,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter
from xngin.apiserver.testing.admin_api_client import AdminAPIClient
from xngin.apiserver.testing.testing_dwh_def import TESTING_DWH_PARTICIPANT_DEF


async def make_experiment(
    xngin_session,
    datasource: tables.Datasource,
    design_spec: DesignSpec,
    *,
    table_name: str | None = None,
    primary_key: str | None = None,
) -> tables.Experiment:
    field_type_map = None
    if design_spec.experiment_type in {ExperimentsType.FREQ_PREASSIGNED, ExperimentsType.FREQ_ONLINE}:
        assert isinstance(design_spec, BaseFrequentistDesignSpec)
        assert table_name is not None
        assert primary_key is not None
        field_type_map = await fetch_fields_or_raise(datasource, design_spec, table_name, primary_key)

    experiment_converter = ExperimentStorageConverter.init_from_components(
        datasource_id=datasource.id,
        organization_id=datasource.organization_id,
        experiment_type=design_spec.experiment_type,
        design_spec=design_spec,
        state=ExperimentState.COMMITTED,
        stopped_assignments_at=datetime.now(UTC),
        stopped_assignments_reason=StopAssignmentReason.PREASSIGNED,
        table_name=table_name,
        field_type_map=field_type_map,
        unique_id_name=primary_key,
    )
    experiment = experiment_converter.get_experiment()
    xngin_session.add(experiment)
    await xngin_session.commit()
    # Add assignments to the experiment so analysis doesn't error.
    arm_assignments = [
        tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_type="",
            participant_id="1",
            arm_id=experiment.arms[0].id,
            strata=[],
        ),
        tables.ArmAssignment(
            experiment_id=experiment.id,
            participant_type="",
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


async def test_make_first_snapshot_of_freq_preassigned(xngin_session, testing_datasource):
    datasource = testing_datasource.ds

    # Create a preassigned frequentist experiment design spec
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
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

    experiment = await make_experiment(
        xngin_session,
        datasource,
        design_spec,
        table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
        primary_key="id",
    )
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
    # Suppress expected statsmodels warnings due to not actually assigning any units to the arms for
    # this test, as we're not focused on the actual analysis.
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=r"(divide by zero|invalid value).*", category=RuntimeWarning)
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
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=r"(divide by zero|invalid value).*", category=RuntimeWarning)
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
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message=r"(divide by zero|invalid value).*", category=RuntimeWarning)
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


async def test_make_first_snapshot_is_noop_when_missing_or_not_pending(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
):
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="missing snapshot test",
        description="missing snapshot test",
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
    experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.ds.id,
        body=CreateExperimentRequest(
            design_spec=design_spec,
            table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
            primary_key="id",
        ),
        desired_n=2,
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=testing_datasource.ds.id, experiment_id=experiment_id)
    completed_snapshot = tables.Snapshot(
        experiment_id=experiment_id,
        status="failed",
        message="already failed",
    )
    xngin_session.add(completed_snapshot)
    await xngin_session.commit()

    def list_snapshots():
        return aclient.list_snapshots(
            organization_id=testing_datasource.org.id,
            datasource_id=testing_datasource.ds.id,
            experiment_id=experiment_id,
        ).data.items

    snapshots_before = list_snapshots()
    assert [snapshot.status for snapshot in snapshots_before] == [SnapshotStatus.FAILED]

    await make_first_snapshot(experiment_id, "sn_missing")
    await make_first_snapshot(experiment_id, completed_snapshot.id)

    snapshots_after = list_snapshots()
    assert [snapshot.status for snapshot in snapshots_after] == [SnapshotStatus.FAILED]
    assert snapshots_after[0].data is None
    assert snapshots_after[0].details == {"message": "already failed"}


async def test_handle_one_snapshot_safely_marks_failed_on_exception(
    testing_datasource, aclient: AdminAPIClient, mocker
):
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="handle snapshot failure test",
        description="handle snapshot failure test",
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
    experiment_id = aclient.create_experiment(
        datasource_id=testing_datasource.ds.id,
        body=CreateExperimentRequest(
            design_spec=design_spec,
            table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
            primary_key="id",
        ),
        desired_n=2,
    ).data.experiment_id
    aclient.commit_experiment(datasource_id=testing_datasource.ds.id, experiment_id=experiment_id)

    # Force the snapshot to fail.
    mocker.patch(
        "xngin.apiserver.snapshots.snapshotter._query_dwh_for_snapshot_data",
        side_effect=RuntimeError("boom"),
    )
    aclient.create_snapshot(
        organization_id=testing_datasource.org.id,
        datasource_id=testing_datasource.ds.id,
        experiment_id=experiment_id,
    )

    snapshots = aclient.list_snapshots(
        organization_id=testing_datasource.org.id,
        datasource_id=testing_datasource.ds.id,
        experiment_id=experiment_id,
    ).data.items
    assert [snapshot.status for snapshot in snapshots] == [SnapshotStatus.FAILED]
    assert snapshots[0].data is None
    assert snapshots[0].details == {"message": "RuntimeError: boom"}


async def test_create_pending_snapshots_inserts_for_new_stale_and_failed_experiments(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
):
    now = datetime.now(UTC)
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="snapshot scheduling test",
        description="snapshot scheduling test",
        start_date=datetime(2024, 1, 1, tzinfo=UTC),
        end_date=now + timedelta(days=1),
        arms=[
            Arm(arm_name="C", arm_description="C"),
            Arm(arm_name="T", arm_description="T"),
        ],
        metrics=[DesignSpecMetricRequest(field_name="is_engaged", metric_pct_change=0.1)],
        strata=[],
        filters=[],
    )

    def create_and_commit_experiment(design_spec: PreassignedFrequentistExperimentSpec) -> str:
        experiment_id = aclient.create_experiment(
            datasource_id=testing_datasource.ds.id,
            body=CreateExperimentRequest(
                design_spec=design_spec,
                table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
                primary_key="id",
            ),
            desired_n=2,
        ).data.experiment_id
        aclient.commit_experiment(datasource_id=testing_datasource.ds.id, experiment_id=experiment_id)
        return experiment_id

    new_experiment_id = create_and_commit_experiment(design_spec)
    stale_experiment_id = create_and_commit_experiment(
        design_spec.model_copy(update={"experiment_name": "stale snapshot test"})
    )
    failed_experiment_id = create_and_commit_experiment(
        design_spec.model_copy(update={"experiment_name": "failed snapshot test"})
    )
    fresh_experiment_id = create_and_commit_experiment(
        design_spec.model_copy(update={"experiment_name": "fresh snapshot test"})
    )
    inactive_experiment_id = create_and_commit_experiment(
        design_spec.model_copy(
            update={
                "experiment_name": "inactive snapshot test",
                "end_date": now - timedelta(days=2),
            }
        )
    )

    xngin_session.add_all([
        tables.Snapshot(
            experiment_id=stale_experiment_id,
            status="success",
            updated_at=now - timedelta(hours=2),
        ),
        tables.Snapshot(
            experiment_id=failed_experiment_id,
            status="failed",
            updated_at=now,
            message="boom",
        ),
        tables.Snapshot(
            experiment_id=fresh_experiment_id,
            status="success",
            updated_at=now - timedelta(minutes=5),
        ),
    ])
    await xngin_session.commit()

    await create_pending_snapshots(3600)

    def list_snapshots(experiment_id: str):
        return aclient.list_snapshots(
            organization_id=testing_datasource.org.id,
            datasource_id=testing_datasource.ds.id,
            experiment_id=experiment_id,
        ).data.items

    assert [snapshot.status for snapshot in list_snapshots(new_experiment_id)] == [SnapshotStatus.RUNNING]

    assert [snapshot.status for snapshot in list_snapshots(stale_experiment_id)] == [
        SnapshotStatus.RUNNING,
        SnapshotStatus.SUCCESS,
    ]

    assert [snapshot.status for snapshot in list_snapshots(failed_experiment_id)] == [
        SnapshotStatus.RUNNING,
        SnapshotStatus.FAILED,
    ]

    assert [snapshot.status for snapshot in list_snapshots(fresh_experiment_id)] == [SnapshotStatus.SUCCESS]

    assert [snapshot.status for snapshot in list_snapshots(inactive_experiment_id)] == []


async def test_process_pending_snapshots_processes_until_empty(
    xngin_session,
    testing_datasource,
    aclient: AdminAPIClient,
):
    design_spec = PreassignedFrequentistExperimentSpec(
        experiment_type=ExperimentsType.FREQ_PREASSIGNED,
        experiment_name="process pending snapshots test",
        description="process pending snapshots test",
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

    experiment_ids = []
    for i in range(2):
        experiment_id = aclient.create_experiment(
            datasource_id=testing_datasource.ds.id,
            body=CreateExperimentRequest(
                design_spec=design_spec.model_copy(update={"experiment_name": f"process pending snapshot test {i}"}),
                table_name=TESTING_DWH_PARTICIPANT_DEF.table_name,
                primary_key="id",
            ),
            desired_n=8,
        ).data.experiment_id
        aclient.commit_experiment(datasource_id=testing_datasource.ds.id, experiment_id=experiment_id)
        experiment_ids.append(experiment_id)

    await create_pending_snapshots(0)

    for experiment_id in experiment_ids:
        snapshots = aclient.list_snapshots(
            organization_id=testing_datasource.org.id,
            datasource_id=testing_datasource.ds.id,
            experiment_id=experiment_id,
        ).data.items
        assert len(snapshots) == 1
        assert snapshots[0].status == SnapshotStatus.RUNNING

    await process_pending_snapshots(SNAPSHOT_TIMEOUT_SECS, max_jitter_secs=0)

    for experiment_id in experiment_ids:
        snapshots = aclient.list_snapshots(
            organization_id=testing_datasource.org.id,
            datasource_id=testing_datasource.ds.id,
            experiment_id=experiment_id,
        ).data.items
        assert len(snapshots) == 1
        assert snapshots[0].status == SnapshotStatus.SUCCESS
        analysis = FreqExperimentAnalysisResponse.model_validate(snapshots[0].data)
        assert isinstance(analysis, FreqExperimentAnalysisResponse)
