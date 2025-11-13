"""Test assignment adapter conversion functions."""

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter
from sqlalchemy import DECIMAL, Boolean, Column, Float, Integer, MetaData, String, Table, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from xngin.apiserver.routers.assignment_adapters import (
    assign_treatments_with_balance,
    bulk_insert_arm_assignments,
    make_balance_check,
)
from xngin.apiserver.routers.common_api_types import (
    Arm,
    BalanceCheck,
    Strata,
)
from xngin.apiserver.routers.experiments.test_experiments_common import (
    insert_experiment_and_arms,
)
from xngin.apiserver.settings import (
    RemoteDatabaseConfig,
)
from xngin.apiserver.sqla import tables
from xngin.stats.assignment import AssignmentResult
from xngin.stats.balance import BalanceResult


class RowProtocolMixin:
    def _asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[no-any-return, call-overload]


@dataclass
class Row(RowProtocolMixin):
    """Simulate the bits of a sqlalchemy Row that we need here."""

    id: int
    age: float
    income: float
    gender: str
    region: str
    skewed: int
    income_dec: Decimal
    is_male: bool
    single_value: int


@pytest.fixture
def sample_table():
    metadata_obj = MetaData()
    return Table(
        "table_name",
        metadata_obj,
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column("age", Integer),
        Column("income", Float),
        Column("gender", String),
        Column("region", String),
        Column("skewed", Float),
        Column("income_dec", DECIMAL),
        Column("is_male", Boolean),
        Column("single_value", Integer),
    )


def make_sample_data_dict(n=1000):
    rs = np.random.default_rng(42)
    data = {
        "id": range(n),
        "age": np.round(rs.normal(30, 5, n), 0),
        "income": np.round(np.float64(rs.lognormal(10, 1, n)), 0),
        "gender": rs.choice(["M", "F"], n),
        "region": rs.choice(["North", "South", "East", "West"], n),
        "skewed": rs.permutation(
            np.concatenate((
                np.repeat([1], int(n * 0.9)),
                np.repeat([0], n - int(n * 0.9)),
            ))
        ),
        "single_value": [1] * n,
    }
    data["income_dec"] = [Decimal(i).quantize(Decimal(1)) for i in data["income"]]
    data["is_male"] = [g == "M" for g in data["gender"]]
    return data


@pytest.fixture(name="sample_data")
def fixture_sample_data():
    """Helper that turns a python dict into a pandas DataFrame."""
    return pd.DataFrame(make_sample_data_dict())


@pytest.fixture(name="sample_rows")
def fixture_sample_rows(sample_data):
    """Helper that turns a pandas DataFrame into a list of SQLAlchemy-like Row objects."""
    return [Row(**row) for row in sample_data.to_dict("records")]


def make_arms(names: list[str]):
    return [Arm(arm_id=tables.arm_id_factory(), arm_name=name) for name in names]


# Tests for the conversion function
def test_make_balance_check():
    """Test conversion from BalanceResult to BalanceCheck."""
    # Test with None input
    assert make_balance_check(None, 0.5) is None

    # Test with actual BalanceResult
    balance_result = BalanceResult(
        f_statistic=1.234567890123456,
        f_pvalue=0.876543210987654,
        model_summary="test summary",
        model_params=[],
        model_param_std_errors=[],
        numerator_df=5.0,
        denominator_df=100.0,
    )
    balance_check = make_balance_check(balance_result, 0.5)

    assert isinstance(balance_check, BalanceCheck)
    assert balance_check.f_statistic == pytest.approx(1.234567890, abs=1e-9)
    assert balance_check.p_value == pytest.approx(0.876543211, abs=1e-9)
    assert balance_check.balance_ok is True
    assert balance_check.numerator_df == 5
    assert balance_check.denominator_df == 100


def test_make_balance_check_with_different_thresholds():
    """Test that balance_ok varies with the threshold."""
    balance_result = BalanceResult(
        f_statistic=2.5,
        f_pvalue=0.3,
        model_summary="test summary",
        model_params=[],
        model_param_std_errors=[],
        numerator_df=3.0,
        denominator_df=50.0,
    )

    balance_check = make_balance_check(balance_result, 0.5)
    assert balance_check is not None

    assert balance_check.balance_ok is False
    assert balance_check.f_statistic == 2.5
    assert balance_check.p_value == 0.3
    assert balance_check.numerator_df == 3
    assert balance_check.denominator_df == 50

    # Try a few other different thresholds
    thresh1 = make_balance_check(balance_result, 1.0)
    assert thresh1 and thresh1.balance_ok is False
    thresh2 = make_balance_check(balance_result, 0.3)
    assert thresh2 and thresh2.balance_ok is False
    thresh3 = make_balance_check(balance_result, 0.299)
    assert thresh3 and thresh3.balance_ok
    thresh4 = make_balance_check(balance_result, 0)
    assert thresh4 and thresh4.balance_ok


def test_assign_treatments_with_balance_basic(sample_table, sample_rows):
    """Test that assign_treatments_with_balance returns proper AssignmentResult."""
    result = assign_treatments_with_balance(
        sa_table=sample_table,
        data=sample_rows,
        stratum_cols=["region", "gender"],
        id_col="id",
        n_arms=2,
        random_state=42,
    )

    # Check AssignmentResult structure
    assert isinstance(result, AssignmentResult)
    assert result.stratum_ids is not None
    assert len(result.stratum_ids) == len(sample_rows)
    assert len(result.treatment_ids) == len(sample_rows)
    assert result.orig_stratum_cols == ["gender", "region"]
    assert result.balance_result is not None
    # Use relative tolerance to accommodate BLAS/LAPACK differences between environments
    # (e.g. Apple Accelerate on macOS vs OpenBLAS on Linux)
    assert result.balance_result.f_statistic == pytest.approx(0.00699, rel=0.3), (
        f"\n{result.balance_result.model_summary}"
    )
    # Although the relative difference looks large, the tiny f-stat is still statistically equivlent
    # to about p≈1 on different platforms.
    assert result.balance_result.f_pvalue == pytest.approx(0.99990, abs=1e-5)


@pytest.mark.parametrize("stratum_id_name", [None, "stratum_id"])
async def test_bulk_insert_arm_assignments_basic(
    xngin_session: AsyncSession,
    testing_datasource,
    sample_rows,
    stratum_id_name: str | None,
):
    """Test bulk inserts of arm assignments, with and without strata group ids."""
    # First create an experiment and arms in db
    ds: tables.Datasource = testing_datasource.ds
    pt = testing_datasource.pt
    experiment = await insert_experiment_and_arms(xngin_session, ds)
    arms = [Arm(arm_id=arm.id, arm_name=arm.name) for arm in experiment.arms]

    # Simulate 2 arms with stratification
    fake_assignment_results = AssignmentResult(
        treatment_ids=[0, 1] * (len(sample_rows) // 2),
        stratum_ids=[int(s.is_male) for s in sample_rows],
        balance_result=None,
        orig_stratum_cols=["gender"],
    )

    await bulk_insert_arm_assignments(
        xngin_session=xngin_session,
        experiment_id=experiment.id,
        arms=arms,
        participant_type=pt.participant_type,
        participant_id_col=pt.get_unique_id_field(),
        data=sample_rows,
        assignment_result=fake_assignment_results,
        stratum_id_name=stratum_id_name,
    )

    # Get assignments for verification
    result = await xngin_session.scalars(select(tables.ArmAssignment))
    assignments = result.all()

    # Verify all participants are assigned
    participant_ids = {a.participant_id for a in assignments}
    expected_ids = {str(row.id) for row in sample_rows}
    assert participant_ids == expected_ids

    # Verify arm assignments
    arm_ids = {arm.arm_id for arm in arms}
    for assignment in assignments:
        assert assignment.experiment_id == experiment.id
        assert assignment.participant_type == pt.participant_type
        assert assignment.arm_id in arm_ids

        # Verify strata are properly stored
        assert assignment.strata[0]["field_name"] == "gender"
        assert assignment.strata[0]["strata_value"] in {"M", "F"}
        assert len(assignment.strata) == 1 if stratum_id_name is None else 2
        if stratum_id_name is not None:
            assert assignment.strata[1]["field_name"] == stratum_id_name
            # Note that all strata values are rendered as strings
            assert assignment.strata[1]["strata_value"] in {"0", "1"}


MAX_SAFE_INTEGER = (1 << 53) - 1  # 9007199254740991


async def test_assign_and_bulk_insert_with_large_integers_as_participant_ids(
    xngin_session: AsyncSession, testing_datasource, sample_table, sample_data
):
    """Test assignment with large integer participant IDs (underlying type as Decimal and int64)."""
    # First create an experiment and arms in db
    ds: tables.Datasource = testing_datasource.ds
    ds_config = TypeAdapter(RemoteDatabaseConfig).validate_python(ds.config)
    pt = ds_config.participants[0]
    experiment = await insert_experiment_and_arms(xngin_session, ds)
    arms = [Arm(arm_id=arm.id, arm_name=arm.name) for arm in experiment.arms]

    async def _assign_test(data):
        rows = [Row(**row) for row in data.to_dict("records")]
        assignment_result = assign_treatments_with_balance(
            sa_table=sample_table,
            data=rows,
            stratum_cols=["gender", "region"],
            id_col="id",
            n_arms=2,
            random_state=42,
        )

        # Bulk insert assignments
        await bulk_insert_arm_assignments(
            xngin_session=xngin_session,
            experiment_id=experiment.id,
            arms=arms,
            participant_type=pt.participant_type,
            participant_id_col="id",
            data=rows,
            assignment_result=assignment_result,
        )

        # Get assignments for verification
        result = await xngin_session.scalars(
            select(tables.ArmAssignment)
            .where(tables.ArmAssignment.experiment_id == experiment.id)
            .order_by(tables.ArmAssignment.participant_id)
        )
        return result.all()

    orig_ids = sample_data["id"].copy()

    # Test: handle Decimals including those bigger than signed int64s
    # (e.g. from psycopg2 with redshift numerics).
    sample_data["id"] = orig_ids.apply(lambda x: Decimal(MAX_SAFE_INTEGER + x))
    assignments = await _assign_test(sample_data)
    # Verify large integer IDs were properly stored as strings
    assert len(assignments) == len(sample_data)
    for assignment in assignments:
        participant_id_int = int(assignment.participant_id)
        assert participant_id_int >= MAX_SAFE_INTEGER
        orig_id = participant_id_int - MAX_SAFE_INTEGER
        # Assert that the inserted id was derived from the original id
        assert orig_id in orig_ids, f"id {orig_id} not found"

    # Reset the assignments
    await xngin_session.execute(delete(tables.ArmAssignment))

    # Test: handle very big negatives as well
    sample_data["id"] = orig_ids.apply(lambda x: Decimal(-MAX_SAFE_INTEGER - x))
    assignments = await _assign_test(sample_data)
    # Verify large integer IDs were properly stored as strings
    assert len(assignments) == len(sample_data)
    for assignment in assignments:
        participant_id_int = int(assignment.participant_id)
        assert participant_id_int <= -MAX_SAFE_INTEGER
        orig_id = -participant_id_int - MAX_SAFE_INTEGER
        # Assert that the inserted id was derived from the original id
        assert orig_id in orig_ids, f"id {orig_id} not found"

    # Reset the assignments
    await xngin_session.execute(delete(tables.ArmAssignment))

    # Test: check that stochatreat isn't upcasting int64 to float64:
    sample_data["id"] = orig_ids.astype("int64")
    # If cast to float64 would round to 9007199254740992
    sample_data.loc[1, "id"] = MAX_SAFE_INTEGER + 2
    # If cast to float64, this next value would be rounded to nonexistent 103241243500726320 and raise a
    # ValueError in our response construction.
    sample_data.loc[2, "id"] = 103241243500726324
    assignments = await _assign_test(sample_data)
    # These raise StopIteration if they don't exist
    next(a for a in assignments if a.participant_id == "9007199254740993")
    next(a for a in assignments if a.participant_id == "103241243500726324")
    ids = {a.participant_id for a in assignments}
    assert ids == set(sample_data["id"].astype(str))


async def test_bulk_insert_renders_decimal_and_bool_strata_correctly(
    xngin_session: AsyncSession, testing_datasource, sample_rows
):
    """Test that the adapter correctly renders decimal and bool strata as strings."""
    # First create an experiment and arms in db
    ds: tables.Datasource = testing_datasource.ds
    pt = testing_datasource.pt
    experiment = await insert_experiment_and_arms(xngin_session, ds)
    arms = [Arm(arm_id=arm.id, arm_name=arm.name) for arm in experiment.arms]

    fake_assignment_results = AssignmentResult(
        treatment_ids=[0, 1] * (len(sample_rows) // 2),
        stratum_ids=[0, 1] * (len(sample_rows) // 2),
        balance_result=None,
        orig_stratum_cols=["income_dec", "is_male"],
    )

    await bulk_insert_arm_assignments(
        xngin_session=xngin_session,
        experiment_id=experiment.id,
        arms=arms,
        participant_type=pt.participant_type,
        participant_id_col=pt.get_unique_id_field(),
        data=sample_rows,
        assignment_result=fake_assignment_results,
    )

    # Get assignments for verification
    result = await xngin_session.scalars(select(tables.ArmAssignment))
    assignments = result.all()

    assert len(assignments) == len(sample_rows)
    for p in assignments:
        # we rounded the Decimal to an int, so shouldn't see the decimal point
        assert len(p.strata) == 2, p.strata
        assert p.strata[0]["field_name"] == "income_dec", p.strata
        assert "." not in p.strata[0]["strata_value"], p.strata
        assert p.strata[1]["field_name"] == "is_male", p.strata
        assert p.strata[1]["strata_value"] in {"True", "False"}, p.strata


async def test_bulk_insert_with_no_stratification(xngin_session: AsyncSession, testing_datasource, sample_rows):
    """Test assignment with no stratification columns."""
    # First create an experiment and arms in db
    ds: tables.Datasource = testing_datasource.ds
    pt = testing_datasource.pt
    experiment = await insert_experiment_and_arms(xngin_session, ds)
    arms = [Arm(arm_id=arm.id, arm_name=arm.name) for arm in experiment.arms]

    fake_assignment_results = AssignmentResult(
        treatment_ids=[0, 1] * (len(sample_rows) // 2),
        stratum_ids=None,
        balance_result=None,
        orig_stratum_cols=[],
    )

    await bulk_insert_arm_assignments(
        xngin_session=xngin_session,
        experiment_id=experiment.id,
        arms=arms,
        participant_type=pt.participant_type,
        participant_id_col=pt.get_unique_id_field(),
        data=sample_rows,
        assignment_result=fake_assignment_results,
    )

    # Get assignments for verification
    result = await xngin_session.scalars(select(tables.ArmAssignment))
    assignments = result.all()

    arm_counts: defaultdict[str, int] = defaultdict(int)
    # There should be no strata in the output
    for p in assignments:
        arm_counts[p.arm_id] += 1
        assert p.strata == []
    # The number of assignments per arm should be equal
    arm0 = arms[0].arm_id
    arm1 = arms[1].arm_id
    assert arm0 is not None and arm1 is not None
    assert arm_counts[arm0] == arm_counts[arm1]
    assert arm_counts[arm0] == len(assignments) // 2


async def test_bulk_insert_with_no_valid_strata(xngin_session: AsyncSession, testing_datasource, sample_rows):
    """Test assignment when a strata column has only a single value."""
    # First create an experiment and arms in db
    ds: tables.Datasource = testing_datasource.ds
    pt = testing_datasource.pt
    experiment = await insert_experiment_and_arms(xngin_session, ds)
    arms = [Arm(arm_id=arm.id, arm_name=arm.name) for arm in experiment.arms]

    # Simulate no stratification case: the strata column only has a single value.
    fake_assignment_results = AssignmentResult(
        treatment_ids=[0, 1] * (len(sample_rows) // 2),
        stratum_ids=None,
        balance_result=None,
        orig_stratum_cols=["single_value"],
    )

    await bulk_insert_arm_assignments(
        xngin_session=xngin_session,
        experiment_id=experiment.id,
        arms=arms,
        participant_type=pt.participant_type,
        participant_id_col=pt.get_unique_id_field(),
        data=sample_rows,
        assignment_result=fake_assignment_results,
    )

    # Get assignments for verification
    result = await xngin_session.scalars(select(tables.ArmAssignment))
    assignments = result.all()

    # Here we still output the requested strata column, even though it's all the same value
    expected_strata = [Strata(field_name="single_value", strata_value="1").model_dump()]
    assert all(p.strata == expected_strata for p in assignments)
