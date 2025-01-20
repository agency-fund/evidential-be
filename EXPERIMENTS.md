# Experiment Config Construction Overview

An experiment config is comprised of the following primary components:
1. An AudienceSpec which defines the pool of potential participants
2. A DesignSpec specifying:
    1. the treatment arms
    2. outcome metrics
    3. strata
    4. experiment meta data (start/end date, name, description)
    5. statistical parameters (power, significance, balance test threshold)
3. Power calculations for the outcome metrics
4. An Assignment for each sampled participant given the AudienceSpec and DesignSpec.

For generating 3 and 4 above, we perform:

1. Baseline data retrieval - mean, stddev (for numeric metrics), and the number of available
   participants targeted by the AudienceSpec
1. Power analysis - Given and AudienceSpec and DesignSpec we analyze the
   statistical power for each metric in the DesignSpec, along with the statistical
   parameters. This occurs as follows for each metric:
    1. If there is no baseline value (and std dev for numeric metrics), go to the
        data source to fetch these values.
    2. Use the declared metric_target or compute this target based on
        metric_pct_change and the baseline value.
    3. Use the number of arms, metric baseline and target, statistical parameters
        and the number of participants available using the Audience filter to determine
        if we're sufficiently powered. If we are not, compute the effect size needed to
        be powered. This power information can be added to the metrics in the DesignSpec.
1. Assignment - the Power analysis computes the minimum number of participants needed
   to be statistically powered "n" which is left to the user to supply for assignment.
   AssignRequest takes the same inputs (AudienceSpec and DesignSpec) to generate a list
   of "n" users randomly assigned using the set of treatment arms and the strata. This
   should return a list of objects containing a participant id, treatment assignment,
   and strata values.
1. Analysis - TBD
