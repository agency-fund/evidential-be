"""Before/after benchmarks for bandit arm-weight to prior-parameter conversion.

Run explicitly with:
    task test -- -m benchmark_bandits -s src/xngin/apiserver/benchmarks/test_bandit_weights_perf.py

bandit_weights_to_normal_prior() integrates a normal density over (-inf, inf) once per arm inside an
objective that scipy.optimize.minimize evaluates dozens of times, so the integrand runs hundreds of times
per conversion. It originally used scipy.stats.norm.pdf/cdf, whose frozen-distribution argument validation
costs far more than the arithmetic it guards; it now uses scipy.special.ndtr and writes the density out
directly.

_legacy_normal_prior() below is that original implementation. It is frozen here as a measurement baseline
and is deliberately NOT kept in sync with the production function: if the production implementation changes
again, this stays as the historical reference point that the speedup is quoted against.

The two parameters that drive the cost map onto POST /datasources/{datasource_id}/experiments as follows.
Participant volume and metric count do not appear: they drive the assignment and power-analysis paths
instead, and are not inputs to this conversion.

    arms=N   len(design_spec.arms), for a bandit spec where every arm sets arm_weight (weights must be
             set on all arms and sum to 100, or get_validated_arm_weights() returns None and no
             conversion happens at all). Bounded by limits.MAX_NUMBER_OF_ARMS = 20.
    dims=D   num_contexts, which storage_format_converters.init_from_components() computes as
             len(design_spec.contexts) or 1. So D is the number of contexts on a CMAB spec, and 1 for
             MABExperimentSpec and MABDwhExperimentSpec, which have no contexts. Bounded by
             limits.MAX_NUMBER_OF_CONTEXTS = 10.
"""

import statistics
import time

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm

from xngin.apiserver.routers.common_api_types import PriorTypes
from xngin.stats.bandit_weights_to_prior import (
    bandit_weights_to_normal_prior,
    convert_arm_weights_to_prior_params,
)

pytestmark = pytest.mark.benchmark_bandits

# Cost grows roughly cubically in the number of arms: the objective integrates once per arm, each integrand
# does work proportional to the arm count, and minimize() needs one finite-difference evaluation per
# parameter per step. The legacy implementation takes ~23s for ten arms, so large cases are timed once.
_ITERATIONS_FOR_LARGE_CASES = 1
_ITERATIONS_FOR_SMALL_CASES = 3
_LARGE_CASE_ARM_COUNT = 8

# The arm weight distributions exercised by test_bandit_weights_to_prior.py, as (weights, num_dimensions),
# plus a ten-arm case. Equal weights short-circuit before the optimizer runs, which is why one case is
# nearly free. MAX_NUMBER_OF_ARMS is 20, so ten arms is a realistic but not worst-case load; twenty arms
# takes ~13s with the current implementation and ~110s with the legacy one, which is too slow to time here.
CASES = [
    ([12.5, 12.5, 25.0, 50.0], 1),
    ([25.0, 75.0], 2),
    ([33.33, 33.33, 33.34], 1),
    ([10.0, 20.0, 30.0, 40.0], 3),
    ([2.5, 5.0, 5.0, 7.5, 7.5, 10.0, 12.5, 15.0, 17.5, 17.5], 1),
]


def _iterations(weights: list[float]) -> int:
    if len(weights) >= _LARGE_CASE_ARM_COUNT:
        return _ITERATIONS_FOR_LARGE_CASES
    return _ITERATIONS_FOR_SMALL_CASES


# The conversion functions scale their argument in place (np.asarray returns the same object for a float64
# array), so every call in these benchmarks gets its own array. Production callers go through
# convert_arm_weights_to_prior_params(), which builds a fresh array from a list.
_MINIMUM_EXPECTED_SPEEDUP = 3.0


def _legacy_normal_prior(expected_probabilities: np.ndarray, num_dimensions: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The scipy.stats.norm implementation of bandit_weights_to_normal_prior(), kept for comparison."""
    expected_probabilities = np.asarray(expected_probabilities, dtype=np.float64)
    expected_probabilities *= 0.01  # Normalize to sum to 1
    sigma_params = np.ones_like(expected_probabilities)
    mu_params = np.zeros_like(expected_probabilities)

    def objective(params: np.ndarray) -> float:
        mus = np.array([*params.tolist(), 0.0])

        def prob_n_is_max(n: int) -> float:
            def integrand(x: float) -> float:
                pdf_n = norm.pdf(x, loc=mus, scale=sigma_params)
                cdf_n = norm.cdf(x, loc=mus, scale=sigma_params)
                return float((np.prod(cdf_n) / (cdf_n[n] + 0.00001)) * pdf_n[n])  # type: ignore[index]

            result, _ = quad(integrand, -np.inf, np.inf)
            return float(result)

        computed_probabilities = np.array([prob_n_is_max(n) for n in range(len(expected_probabilities))])
        return float(
            np.sum(
                (computed_probabilities**num_dimensions - expected_probabilities) ** 2 + 0.01 * num_dimensions * mus**2
            )
        )

    if (expected_probabilities.round(1) == expected_probabilities[0].round(1)).all():
        return mu_params, sigma_params
    result = minimize(objective, mu_params[:-1])
    return np.array([*result.x.tolist(), 0.0]), sigma_params


def _time_conversion(fn, weights: list[float], num_dimensions: int) -> list[float]:
    """Times fn, giving each run its own input array."""
    timings = []
    for _ in range(_iterations(weights)):
        arg = np.array(weights, dtype=np.float64)
        started = time.perf_counter()
        fn(arg, num_dimensions=num_dimensions)
        timings.append(time.perf_counter() - started)
    return timings


def _summary_line(name: str, timings: list[float]) -> str:
    return (
        f"BENCH {name}: n={len(timings)} "
        f"min={min(timings):.4f}s median={statistics.median(timings):.4f}s "
        f"mean={statistics.mean(timings):.4f}s max={max(timings):.4f}s"
    )


@pytest.mark.parametrize(("weights", "num_dimensions"), CASES)
def test_normal_prior_matches_legacy_implementation(weights: list[float], num_dimensions: int):
    """The faster implementation must agree with the one it replaced.

    The two agreed bit-for-bit when this was written, because scipy.stats.norm.cdf defers to ndtr and its
    pdf uses the same closed form; the tolerance here only guards against platform and scipy differences.
    """
    legacy_mu, legacy_sigma = _legacy_normal_prior(np.array(weights, dtype=np.float64), num_dimensions=num_dimensions)
    mu, sigma = bandit_weights_to_normal_prior(np.array(weights, dtype=np.float64), num_dimensions=num_dimensions)

    np.testing.assert_allclose(mu, legacy_mu, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(sigma, legacy_sigma, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(("weights", "num_dimensions"), CASES)
def test_normal_prior_is_faster_than_legacy_implementation(weights: list[float], num_dimensions: int):
    """Reports before/after timings per case and fails if the speedup regresses badly.

    The floor is deliberately far below the ~9x observed when this was written so that a loaded machine
    does not fail the build; the printed numbers are the useful output.
    """
    label = f"arms={len(weights)},dims={num_dimensions}"
    legacy_timings = _time_conversion(_legacy_normal_prior, weights, num_dimensions)
    current_timings = _time_conversion(bandit_weights_to_normal_prior, weights, num_dimensions)

    print(_summary_line(f"normal_prior_legacy {label}", legacy_timings))
    print(_summary_line(f"normal_prior_current {label}", current_timings))

    # Equal weights short-circuit before the optimizer runs, so there is no integration to speed up.
    if len(set(np.round(weights, 1))) == 1:
        pytest.skip("equal arm weights short-circuit before the optimizer runs")

    speedup = min(legacy_timings) / min(current_timings)
    print(f"BENCH normal_prior_speedup {label}: {speedup:.1f}x")
    assert speedup >= _MINIMUM_EXPECTED_SPEEDUP, (
        f"expected at least {_MINIMUM_EXPECTED_SPEEDUP}x over the scipy.stats implementation, got {speedup:.1f}x"
    )


def test_convert_arm_weights_to_prior_params_totals():
    """Times the entry point the experiment-creation request path actually calls, for both prior types."""
    for prior_type in (PriorTypes.BETA, PriorTypes.NORMAL):
        timings = []
        for _ in range(_ITERATIONS_FOR_SMALL_CASES):
            started = time.perf_counter()
            convert_arm_weights_to_prior_params([25.0, 75.0], prior_type=prior_type, num_contexts=2)
            timings.append(time.perf_counter() - started)
        print(_summary_line(f"convert_arm_weights_to_prior_params[{prior_type.value}]", timings))
