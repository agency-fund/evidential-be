import pandas as pd
import statsmodels.formula.api as smf
from patsy import EvalFactor
from xngin.apiserver.api_types import (
    Assignment,
    ExperimentAnalysis,
    ParticipantOutcome,
)


def analyze_experiment(
    treatment_assignments: list[Assignment],
    participant_outcomes: list[ParticipantOutcome],
) -> list[ExperimentAnalysis]:
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
            "arm_name": assignment.arm_name,
        }
        for assignment in treatment_assignments
    ])

    analyses = []

    for i in range(len(participant_outcomes[0].metric_values)):
        metric_name = participant_outcomes[0].metric_values[i].metric_name
        outcomes_df = pd.DataFrame([
            {
                "participant_id": outcome.participant_id,
                "metric_value": outcome.metric_values[i].metric_value,
            }
            for outcome in participant_outcomes
        ])
        outcomes_df = outcomes_df.rename(columns={"metric_value": metric_name})
        merged_df = assignments_df.merge(outcomes_df, on="participant_id", how="left")
        model = smf.ols(f"{metric_name} ~ arm_id", data=merged_df).fit()
        arm_names = model.model.data.design_info.factor_infos[
            EvalFactor("arm_id")
        ].categories

        analyses.append(
            ExperimentAnalysis(
                metric_name=metric_name,
                arm_ids=arm_names,
                coefficients=model.params,
                pvalues=model.pvalues,
                tstats=model.tvalues,
                std_errors=list(model.bse),
            )
        )

    return analyses
