import numpy as np
import statsmodels.stats.api as sms
from typing import Dict, Union, Tuple, List
from dataclasses import dataclass
from enum import Enum

class MetricType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"

@dataclass
class MetricSpec:
    """Specification for a metric to analyze."""
    metric_name: str
    metric_type: MetricType
    metric_baseline: float
    metric_target: float = None
    metric_pct_change: float = None
    metric_stddev: float = None
    metric_available_n: int = None

@dataclass
class MetricAnalysis:
    """Analysis results for a single metric."""
    metric_name: str
    metric_type: MetricType
    metric_baseline: float
    metric_target: float
    available_n: int
    target_n: int = None
    sufficient_n: bool = None
    needed_target: float = None
    metric_target_possible: float = None
    metric_pct_change_possible: float = None
    delta: float = None
    msg: str = None

def analyze_metric_power(
    metric: MetricSpec,
    n_arms: int,
    power: float = 0.8,
    alpha: float = 0.05
) -> MetricAnalysis:
    """
    Analyze power for a single metric.
    
    Args:
        metric: MetricSpec containing metric details
        n_arms: Number of treatment arms
        power: Desired statistical power
        alpha: Significance level
    
    Returns:
        MetricAnalysis containing power analysis results
    """
    analysis = MetricAnalysis(
        metric_name=metric.metric_name,
        metric_type=metric.metric_type,
        metric_baseline=metric.metric_baseline,
        metric_target=metric.metric_target,
        available_n=metric.metric_available_n
    )

    # Case A: Both target and baseline defined - calculate required n
    if metric.metric_target is not None and metric.metric_baseline is not None:
        if metric.metric_type == MetricType.CONTINUOUS:
            power_analysis = sms.TTestIndPower()
            target_n = np.ceil(
                power_analysis.solve_power(
                    effect_size=(metric.metric_target - metric.metric_baseline) / metric.metric_stddev,
                    alpha=alpha,
                    power=power,
                    ratio=1
                )
            ) * n_arms
        else:  # BINARY
            power_analysis = sms.TTestIndPower()
            effect_size = sms.proportion_effectsize(
                metric.metric_baseline,
                metric.metric_target
            )
            target_n = np.ceil(
                power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1
                )
            ) * n_arms

        analysis.target_n = int(target_n)
        analysis.sufficient_n = bool(target_n <= metric.metric_available_n)

        msg = f"there are {metric.metric_available_n} units available to run your experiment and {target_n} units are needed to meet your experimental design specs."

        if analysis.sufficient_n:
            msg += f" there are enough units available, you only need {target_n} of the {metric.metric_available_n} units to meet your experimental design specs."
        else:
            # Calculate needed target if insufficient sample
            if metric.metric_type == MetricType.CONTINUOUS:
                power_analysis = sms.TTestIndPower()
                needed_delta = power_analysis.solve_power(
                    nobs1=metric.metric_available_n // n_arms,
                    effect_size=None,
                    alpha=alpha,
                    power=power
                ) * metric.metric_stddev
                needed_target = needed_delta + metric.metric_baseline
            else:  # BINARY
                power_analysis = sms.NormalIndPower()
                # Calculate minimum detectable effect size given sample size
                min_effect_size = power_analysis.solve_power(
                    nobs1=metric.metric_available_n // n_arms,
                    alpha=alpha,
                    power=power,
                    ratio=1
                )
                
                # Convert Cohen's h back to proportion
                # h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
                # where p1 is baseline and p2 is target
                p1 = metric.metric_baseline
                arcsin_p2 = 2 * np.arcsin(np.sqrt(p1)) - min_effect_size
                needed_target = np.sin(arcsin_p2 / 2) ** 2

            analysis.needed_target = needed_target
            msg += (f" there are not enough units available, you need {target_n - metric.metric_available_n} more units "
                   f"to meet your experimental design specifications. in order to meet your specification with the available "
                   f"{metric.metric_available_n} units and a baseline metric value of {metric.metric_baseline:.4f}, your metric "
                   f"target value needs to be {needed_target:.4f}, the current target is {metric.metric_target:.4f}.")

        analysis.msg = msg

    # Case B: Only baseline defined - calculate possible effect size
    elif metric.metric_baseline is not None and metric.metric_target is None:
        if metric.metric_type == MetricType.CONTINUOUS:
            power_analysis = sms.TTestIndPower()
            delta = power_analysis.solve_power(
                nobs1=metric.metric_available_n // n_arms,
                effect_size=None,
                alpha=alpha,
                power=power
            ) * metric.metric_stddev

            analysis.metric_target_possible = delta + metric.metric_baseline
            analysis.metric_pct_change_possible = delta / metric.metric_baseline
            analysis.delta = delta

        else:  # BINARY
            power_analysis = sms.proportion_effectsize(metric.metric_baseline, None)
            p2 = power_analysis.solve_power(
                n=metric.metric_available_n // n_arms,
                effect_size=None,
                alpha=alpha,
                power=power
            )

            analysis.metric_target_possible = p2
            analysis.metric_pct_change_possible = (p2 - metric.metric_baseline) / metric.metric_baseline
            analysis.delta = p2 - metric.metric_baseline

        analysis.msg = (f"the smallest detectable effect size for the given specifications is: {analysis.delta:.4f}pp "
                       f"using all {analysis.available_n} units available to run your experiment. This is a "
                       f"{analysis.metric_pct_change_possible:.1%} increase on the baseline value of {metric.metric_baseline:.4f}.")

    else:
        analysis.msg = "Could not calculate metric baseline with given specification. Provide metric baseline or adjust filters."

    return analysis

def check_power(
    metrics: List[MetricSpec],
    n_arms: int,
    power: float = 0.8,
    alpha: float = 0.05
) -> List[MetricAnalysis]:
    """
    Check power for multiple metrics.
    
    Args:
        metrics: List of MetricSpec objects to analyze
        n_arms: Number of treatment arms
        power: Desired statistical power
        alpha: Significance level
    
    Returns:
        List of MetricAnalysis results
    """
    return [
        analyze_metric_power(metric, n_arms, power, alpha)
        for metric in metrics
    ] 