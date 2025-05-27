import dataclasses
import pandas as pd
import statsmodels.formula.api as smf

from patsy.eval import EvalFactor
from xngin.apiserver.routers.stateless_api_types import (
    ParticipantOutcome,
)
from xngin.apiserver.models import tables


@dataclasses.dataclass(slots=True)  # slots=True for performance
class ArmAnalysisResult:
    is_baseline: bool
    estimate: float
    p_value: float
    t_stat: float
    std_error: float


def analyze_experiment(
    treatment_assignments: list[tables.ArmAssignment],
    participant_outcomes: list[ParticipantOutcome],
    baseline_arm_id: str | None = None,
) -> dict[str, dict[str, ArmAnalysisResult]]:
    """
    Perform statistical analysis with DesignSpec metrics and their values

    Args:
    treatment_assignments: list of participant treatment assignments
    participant_outcomes: list of participant outcomes
    baseline_arm_id: which arm to use as baseline; if not provided, uses the first arm seen

    Returns: map of metric name => map of arm_id => analysis results
    """

    assignments_df = pd.DataFrame([
        {
            "participant_id": assignment.participant_id,
            "arm_id": assignment.arm_id,
        }
        for assignment in treatment_assignments
    ])

    rows = []
    for outcome in participant_outcomes:
        data_row: dict[str, float | str] = {"participant_id": outcome.participant_id}
        for metric_value in outcome.metric_values:
            data_row[metric_value.metric_name] = metric_value.metric_value
        rows.append(data_row)
    outcomes_df = pd.DataFrame(rows)

    merged_df = assignments_df.merge(outcomes_df, on="participant_id", how="left")

    # Make arm_id categorical and ensure baseline_arm_id is first in the categories
    merged_df["arm_id"] = pd.Categorical(merged_df["arm_id"])
    if baseline_arm_id in merged_df["arm_id"].cat.categories:
        arm_ids = merged_df["arm_id"].cat.categories.tolist()
        arm_ids.remove(baseline_arm_id)
        arm_ids.insert(0, baseline_arm_id)
        merged_df["arm_id"] = merged_df["arm_id"].cat.reorder_categories(arm_ids)

    metric_analyses: dict[str, dict[str, ArmAnalysisResult]] = {}
    for metric_name in merged_df.columns:
        if metric_name in {"arm_id", "participant_id"}:
            continue
        model = smf.ols(f"{metric_name} ~ arm_id", data=merged_df).fit()
        arm_ids = model.model.data.design_info.factor_infos[
            EvalFactor("arm_id")
        ].categories
        arm_analyses: dict[str, ArmAnalysisResult] = {}
        for i, arm_id in enumerate(arm_ids):
            arm_analyses[arm_id] = ArmAnalysisResult(
                is_baseline=i == 0
                if baseline_arm_id is None
                else arm_id == baseline_arm_id,
                estimate=float(model.params.iloc[i]),
                p_value=float(model.pvalues.iloc[i]),
                t_stat=float(model.tvalues.iloc[i]),
                std_error=float(list(model.bse)[i]),
            )
        metric_analyses[metric_name] = arm_analyses
    return metric_analyses
