import numpy as np
from loguru import logger
from sqlalchemy import select, text
from sqlalchemy.orm import Session, selectinload

from xngin.apiserver.routers.common_api_types import ExperimentAnalysisResponse
from xngin.apiserver.routers.common_enums import ContextType, ExperimentsType
from xngin.apiserver.routers.experiments import experiments_common
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_format_converters import ExperimentStorageConverter


def capture_snapshot(session: Session, experiment_id: str, *, dwh_timeout: float) -> tables.Snapshot:
    """Record a successful or failed analysis in the caller's transaction.

    dwh_timeout bounds the analysis's data warehouse read.
    """
    try:
        # Analysis is best-effort. A savepoint also recovers from SQL errors so the failure record
        # can commit in the owning transaction without releasing the experiment's maintenance locks.
        with session.begin_nested():
            experiment = session.scalars(
                select(tables.Experiment)
                .where(tables.Experiment.id == experiment_id)
                .options(
                    selectinload(tables.Experiment.arms),
                    selectinload(tables.Experiment.contexts),
                    selectinload(tables.Experiment.datasource),
                    selectinload(tables.Experiment.experiment_fields).selectinload(
                        tables.ExperimentField.experiment_filters
                    ),
                )
            ).one()
            analysis = _compute_snapshot_data(session, experiment, dwh_timeout)
            data = analysis.model_dump(mode="json")
    except Exception as exc:
        # Snapshot task boundary: report an analysis failure to users, but never reconnect and write
        # after losing the transaction that owns this experiment.
        if not session.is_active or session.connection().invalidated:
            raise
        logger.opt(exception=exc).info("Snapshot analysis failed.")
        snapshot = tables.Snapshot(experiment_id=experiment_id, status="failed", message=f"{type(exc).__name__}: {exc}")
    else:
        snapshot = tables.Snapshot(experiment_id=experiment_id, status="success", data=data)
    session.add(snapshot)
    session.flush()
    return snapshot


def _compute_snapshot_data(
    session: Session, experiment: tables.Experiment, dwh_timeout: float
) -> ExperimentAnalysisResponse:
    """Analyze one experiment for its snapshot. dwh_timeout bounds any data warehouse read."""
    experiment_type = ExperimentsType(experiment.experiment_type)
    if experiment_type.is_bandit():
        # Bandit analysis reads only the application database, so dwh_timeout has no warehouse
        # interaction here to bound. That covers MAB_ONLINE_DWH too: its warehouse read happens in
        # the worker's DWH pull job.
        context_vals = None

        # TODO: If the experiment is a CMAB, we need to pass in context values.
        # Ideally we should marginalize over contexts, but we'll start with just using
        # the mean context values. Captured in issue 140
        # (https://github.com/agency-fund/evidential-sprint/issues/140)
        if experiment_type.is_cmab():
            contexts = experiment.contexts
            sorted_contexts = sorted(contexts, key=lambda c: c.id)
            # draw.context_vals were already sorted corresponding to the sorted_contexts
            # ordering when the assignment was made.
            mean_context_vals = _mean_context_vals_from_draws(session, experiment.id, len(contexts))
            context_vals = [
                abs(float(np.ceil(m - 0.5))) if context.value_type == ContextType.BINARY else m
                for m, context in zip(mean_context_vals, sorted_contexts, strict=True)
            ]

        return experiments_common.analyze_experiment_bandit_impl(
            xngin_session=session, experiment=experiment, context_vals=context_vals
        )

    if experiment_type.is_freq():
        # Look for the arm in position 1. If not found, use the first arm.
        baseline_arm = next((arm for arm in experiment.arms if arm.position == 1), experiment.arms[0])
        return experiments_common.analyze_experiment_freq_impl(
            xngin_session=session,
            dsconfig=experiment.datasource.get_config(),
            experiment=experiment,
            baseline_arm_id=baseline_arm.id,
            metrics=ExperimentStorageConverter(experiment).get_design_spec_metrics(),
            dwh_timeout=dwh_timeout,
        )
    raise ValueError(f"Unsupported experiment type: {experiment_type}")


def _mean_context_vals_from_draws(session: Session, experiment_id: str, n_contexts: int) -> list[float]:
    """Per-position mean of non-null context_vals across draws.

    Returns a zero vector when no rows match. Assumes all non-null context_vals
    arrays share the same length and contain no inner NULLs — both invariants
    enforced by the assignment write path.
    """
    if n_contexts == 0:
        return []
    rows = (
        session.execute(
            text("""
                SELECT AVG(val)
                FROM draws, unnest(context_vals) WITH ORDINALITY AS t(val, ord)
                WHERE experiment_id = :eid AND context_vals IS NOT NULL
                GROUP BY ord
                ORDER BY ord
            """),
            {"eid": experiment_id},
        )
    ).all()
    if not rows:
        return [0.0] * n_contexts
    return [r[0] for r in rows]
