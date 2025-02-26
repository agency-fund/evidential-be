import pandas as pd
from pandas import DataFrame
import numpy as np

import statsmodels.formula.api as smf
from xngin.apiserver.api_types import (
    Assignment,
    ExperimentAnalysis,
    ParticipantOutcome,
)


def analyze_experiment(
    treatmentAssignments: list[Assignment],
    participantOutcomes: list[ParticipantOutcome]
) -> ExperimentAnalysis:
    """
    Perform statistical analysis with DesignSpec metrics and their values
    
    Args:
    treatmentAssignments: list of participant treatment assignments
    participantOutcomes: list of participant outcomes
    """

    assignments_df = DataFrame([assignment._asdict() for assignment in treatmentAssignments])
    outcomes_df = DataFrame([outcome._asdict() for outcome in participantOutcomes])
    merged_df = pd.merge(assignments_df, outcomes_df, on='participant_id', how='left')

    model = smf.ols('metric_value ~ C(arm_id)', data=merged_df).fit()

    results = ExperimentAnalysis(
        arm_ids=list(set(merged_df['arm_id'])),
        coefficients=model.params,
        pvalues=model.pvalues,
        tstats=model.tvalues
    )

    return(results)