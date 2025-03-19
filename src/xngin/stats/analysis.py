import pandas as pd
import statsmodels.formula.api as smf
from patsy import EvalFactor
from xngin.apiserver.api_types import (
    ParticipantOutcome,
)
from xngin.apiserver.models.tables import ArmAssignment


def analyze_experiment(
    treatment_assignments: list[ArmAssignment],
    participant_outcomes: list[ParticipantOutcome],
) -> dict[dict]:
    """
    Perform statistical analysis with DesignSpec metrics and their values

    Args:
    treatment_assignments: list of participant treatment assignments
    participant_outcomes: list of participant outcomes
    """

    assignments_df = pd.DataFrame([
        {
            "participant_id": assignment.participant_id,
            "arm_id": assignment.arm_id,
        }
        for assignment in treatment_assignments
    ])

    outcomes_df = pd.DataFrame()
    rows = []
    for outcome in participant_outcomes:
        data_row = {"participant_id": outcome.participant_id}
        for metric_value in outcome.metric_values:
            data_row[metric_value.metric_name] = metric_value.metric_value
        rows.append(data_row)
    outcomes_df = pd.DataFrame(rows)

    merged_df = assignments_df.merge(outcomes_df, on="participant_id", how="left")

    metric_analyses = {}
    for metric_name in merged_df.columns:
        if metric_name in ("arm_id", "participant_id"):
            continue
        model = smf.ols(f"{metric_name} ~ arm_id", data=merged_df).fit()
        arm_ids = model.model.data.design_info.factor_infos[
            EvalFactor("arm_id")
        ].categories
        arm_analyses = {}
        for i in range(len(arm_ids)):
            arm_analyses[arm_ids[i]] = {
                # TODO(roboton): Fix this once we implement #299
                "is_baseline": i == 0,
                "estimate": model.params.iloc[i],
                "p_value": model.pvalues.iloc[i],
                "t_stat": model.tvalues.iloc[i],
                "std_error": list(model.bse)[i],
            }
        metric_analyses[metric_name] = arm_analyses
    return metric_analyses
