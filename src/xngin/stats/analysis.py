import pandas as pd

import statsmodels.formula.api as smf
from xngin.apiserver.api_types import (
    Assignment,
    ExperimentAnalysis,
    ParticipantOutcome,
)


def analyze_experiment(
    treatment_assignments: list[Assignment],
    participant_outcomes: list[ParticipantOutcome],
) -> ExperimentAnalysis:
    """
    Perform statistical analysis with DesignSpec metrics and their values

    Args:
    treatment_assignments: list of participant treatment assignments
    participanta_outcomes: list of participant outcomes
    """

    assignments_df = pd.DataFrame([
        {
            "participant_id": assignment.participant_id,
            "arm_id": assignment.arm_id,
            "arm_name": assignment.arm_name,
        }
        for assignment in treatment_assignments
    ])

    outcomes_df = pd.DataFrame([
        {"participant_id": outcome.participant_id, "metric_value": outcome.metric_value}
        for outcome in participant_outcomes
    ])

    merged_df = assignments_df.merge(outcomes_df, on="participant_id", how="left")

    model = smf.ols("metric_value ~ C(arm_id)", data=merged_df).fit()

    return ExperimentAnalysis(
        arm_ids=list(set(merged_df["arm_id"])),
        coefficients=model.params,
        pvalues=model.pvalues,
        tstats=model.tvalues,
    )
