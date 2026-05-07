"""Converts between API and jsonb storage models used by our internal database models.

To better decouple the API from the storage models, this file defines helpers to set/get JSONB
columns of their respective SQLALchemy models, and construct API types from SQLA/jsonb storage
types. Our SQLA tables ideally shouldn't depend on xngin/apiserver/*; but for those that declare
JSONB type columns for multi-value/complex types, use the converters to get/set them properly.
"""

import operator
from datetime import datetime
from typing import Any, assert_never

import numpy as np
from pydantic import TypeAdapter

from xngin.apiserver.routers import common_api_types as capi
from xngin.apiserver.routers.common_enums import (
    DataType,
    DataTypeStorageClass,
    ExperimentState,
    ExperimentsType,
    StopAssignmentReason,
)
from xngin.apiserver.sqla import tables
from xngin.stats.bandit_weights_to_prior import (
    convert_arm_weights_to_prior_params,
)


def _set_experiment_fields_from_design_spec(
    experiment: tables.Experiment,
    design_spec: capi.DesignSpec,
    field_type_map: dict[str, DataType] | None = None,
) -> None:
    """Save the field-related components of a DesignSpec to an experiment."""
    match design_spec:
        case capi.MABExperimentSpec() | capi.CMABExperimentSpec() | capi.BayesABExperimentSpec():
            experiment.design_spec_fields = None
            experiment.experiment_fields = []
            return
        case capi.PreassignedFrequentistExperimentSpec() | capi.OnlineFrequentistExperimentSpec():
            pass
        case _:
            assert_never(design_spec)

    field_type_map = field_type_map or {}
    unique_id_name = design_spec.primary_key

    # Clear existing design fields
    experiment.experiment_fields = []

    # New fields used in the experiment. Each key is a field name and maps to a ExperimentField object.
    fields_used_map: dict[str, tables.ExperimentField] = {}

    fields_used_map[unique_id_name] = tables.ExperimentField(
        field_name=unique_id_name,
        data_type=field_type_map.get(unique_id_name, DataType.UNKNOWN).value,
        is_unique_id=True,
        experiment_filters=[],
    )

    # Add filters. Fields used as filters technically could be reused with different filter values.
    for idx, filter_item in enumerate(design_spec.filters):
        field = fields_used_map.get(filter_item.field_name)
        datatype = field_type_map.get(filter_item.field_name, DataType.UNKNOWN)
        # Create the new field if it doesn't exist
        if field is None:
            field = tables.ExperimentField(
                field_name=filter_item.field_name,
                data_type=datatype.value,
                experiment_filters=[],
            )
            fields_used_map[filter_item.field_name] = field

        # and associate new filter metadata with the field
        filters = field.experiment_filters or []
        values = filter_item.value
        match datatype.storage_class():
            case DataTypeStorageClass.BOOLEAN:
                values = [None if v is None else bool(v) for v in values]
                filters.append(
                    tables.ExperimentFilter(
                        position=idx + 1,
                        relation=filter_item.relation,
                        boolean_values=values,
                    )
                )
            case DataTypeStorageClass.NUMERIC:
                filters.append(
                    tables.ExperimentFilter(
                        position=idx + 1,
                        relation=filter_item.relation,
                        numeric_values=values,
                    )
                )
            case _:
                values = [None if v is None else str(v) for v in values]
                filters.append(
                    tables.ExperimentFilter(
                        position=idx + 1,
                        relation=filter_item.relation,
                        string_values=values,
                    )
                )

        field.experiment_filters = filters

    # Add metrics
    if design_spec.metrics:
        for index, metric in enumerate(design_spec.metrics):
            field = fields_used_map.get(metric.field_name)
            if field is None:
                field = tables.ExperimentField(
                    field_name=metric.field_name,
                    data_type=field_type_map.get(metric.field_name, DataType.UNKNOWN).value,
                    experiment_filters=[],
                )
                fields_used_map[metric.field_name] = field

            if index == 0:
                field.is_primary_metric = True
            field.metric_pct_change = metric.metric_pct_change
            field.metric_target = metric.metric_target

    # Add strata
    if design_spec.strata:
        for stratum in design_spec.strata:
            field = fields_used_map.get(stratum.field_name)
            if field is None:
                field = tables.ExperimentField(
                    field_name=stratum.field_name,
                    data_type=field_type_map.get(stratum.field_name, DataType.UNKNOWN).value,
                    experiment_filters=[],
                )
                fields_used_map[stratum.field_name] = field

            field.is_strata = True

    # add fields_used_map to experiment_fields
    experiment.experiment_fields = list(fields_used_map.values())


def _set_balance_check_json(experiment: tables.Experiment, value: capi.BalanceCheck | None) -> None:
    if value is None:
        experiment.balance_check = None
    else:
        experiment.balance_check = capi.BalanceCheck.model_validate(value).model_dump()


def _set_power_response_json(experiment: tables.Experiment, value: capi.PowerResponse | None) -> None:
    if value is None:
        experiment.power_analyses = None
    else:
        experiment.power_analyses = capi.PowerResponse.model_validate(value).model_dump()


def experiment_from_design_spec(
    *,
    datasource_id: str,
    organization_id: str,
    design_spec: capi.DesignSpec,
    state: ExperimentState = ExperimentState.ASSIGNED,
    stopped_assignments_at: datetime | None = None,
    stopped_assignments_reason: StopAssignmentReason | str | None = None,
    balance_check: capi.BalanceCheck | None = None,
    power_analyses: capi.PowerResponse | None = None,
    n_trials: int = 0,
    decision: str = "",
    impact: str = "",
    field_type_map: dict[str, DataType] | None = None,
    participant_type: str = "",
) -> tables.Experiment:
    """Create an Experiment ORM object from a design spec and storage metadata."""
    datasource_table = (
        design_spec.table_name
        if isinstance(design_spec, capi.PreassignedFrequentistExperimentSpec | capi.OnlineFrequentistExperimentSpec)
        else None
    )

    # Initialize common fields
    experiment = tables.Experiment(
        datasource_id=datasource_id,
        experiment_type=design_spec.experiment_type,
        participant_type=participant_type,
        datasource_table=datasource_table,
        name=design_spec.experiment_name,
        description=design_spec.description,
        design_url=str(design_spec.design_url) if design_spec.design_url else None,
        state=state.value,
        start_date=design_spec.start_date,
        end_date=design_spec.end_date,
        stopped_assignments_at=stopped_assignments_at,
        stopped_assignments_reason=stopped_assignments_reason,
        decision=decision,
        impact=impact,
    )

    match design_spec:
        case capi.PreassignedFrequentistExperimentSpec() | capi.OnlineFrequentistExperimentSpec():
            # Set frequentist-specific fields
            experiment.power = design_spec.power
            experiment.alpha = design_spec.alpha
            experiment.fstat_thresh = design_spec.fstat_thresh

            experiment.arms = [
                tables.Arm(
                    name=arm.arm_name,
                    description=arm.arm_description,
                    arm_weight=arm.arm_weight,
                    position=i,
                    experiment_id=experiment.id,
                    organization_id=organization_id,
                )
                for i, arm in enumerate(design_spec.arms, start=1)
            ]
            _set_experiment_fields_from_design_spec(experiment, design_spec, field_type_map)
            _set_balance_check_json(experiment, balance_check)
            _set_power_response_json(experiment, power_analyses)
            return experiment

        case capi.MABExperimentSpec() | capi.CMABExperimentSpec():
            if isinstance(design_spec, capi.CMABExperimentSpec) and not design_spec.contexts:
                raise ValueError(f"CMAB experiment {experiment.id} must have contexts set.")

            # Set bandit fields
            context_len = len(design_spec.contexts) if design_spec.contexts else 1
            if design_spec.contexts:
                experiment.contexts = [
                    tables.Context(
                        name=context.context_name,
                        description=context.context_description,
                        value_type=context.value_type.value,
                        experiment_id=experiment.id,
                    )
                    for context in design_spec.contexts
                ]
            experiment.reward_type = design_spec.reward_type.value
            experiment.prior_type = design_spec.prior_type.value
            experiment.n_trials = n_trials

            arm_weights = design_spec.get_validated_arm_weights()
            if arm_weights:
                # TODO: this method can be expensive and should be on a thread.
                param1, param2 = convert_arm_weights_to_prior_params(
                    arm_weights=arm_weights,
                    prior_type=design_spec.prior_type,
                    num_contexts=context_len,
                )
                match design_spec.prior_type:
                    case capi.PriorTypes.BETA:
                        for arm, alpha, beta in zip(design_spec.arms, param1, param2, strict=True):
                            arm.alpha_init = alpha
                            arm.beta_init = beta
                    case capi.PriorTypes.NORMAL:
                        for arm, mu, sigma in zip(design_spec.arms, param1, param2, strict=True):
                            arm.mu_init = mu
                            arm.sigma_init = sigma

            experiment.arms = [
                tables.Arm(
                    name=arm.arm_name,
                    description=arm.arm_description,
                    arm_weight=arm.arm_weight,
                    position=i,
                    experiment_id=experiment.id,
                    organization_id=organization_id,
                    mu_init=arm.mu_init,
                    sigma_init=arm.sigma_init,
                    mu=None if arm.mu_init is None else [arm.mu_init] * context_len,
                    covariance=None if arm.sigma_init is None else np.diag([arm.sigma_init] * context_len).tolist(),
                    alpha_init=arm.alpha_init,
                    beta_init=arm.beta_init,
                    alpha=arm.alpha_init,
                    beta=arm.beta_init,
                )
                for i, arm in enumerate(design_spec.arms, start=1)
            ]

            return experiment
        case capi.BayesABExperimentSpec():
            raise ValueError(f"Unsupported design_spec type: {type(design_spec)}.")
        case _:
            assert_never(design_spec)


def _convert_experiment_field_to_api_filters(
    experiment_field: tables.ExperimentField,
) -> list[tuple[int, capi.Filter]]:
    """Convert an ExperimentField to (position, API filter) tuples, or an empty list if not a filter."""
    filters = []
    for f in experiment_field.experiment_filters or []:
        values: list[Any]
        if experiment_field.data_type == DataType.BIGINT:
            values = [None if v is None else str(v) for v in (f.numeric_values or [])]
        else:
            values = f.string_values or f.numeric_values or f.boolean_values or []
        filters.append((
            f.position,
            capi.Filter(
                field_name=experiment_field.field_name,
                relation=capi.Relation(f.relation),
                value=values,
            ),
        ))

    return filters


def _convert_experiment_field_to_api_metric(
    experiment_field: tables.ExperimentField,
) -> capi.DesignSpecMetricRequest | None:
    """Convert an ExperimentField to an API metric, or None if the field is not a metric."""
    if experiment_field.is_metric:
        return capi.DesignSpecMetricRequest(
            field_name=experiment_field.field_name,
            metric_pct_change=experiment_field.metric_pct_change,
            metric_target=experiment_field.metric_target,
        )
    return None


def _convert_experiment_field_to_api_stratum(
    experiment_field: tables.ExperimentField,
) -> capi.Stratum | None:
    """Convert an ExperimentField to an API stratum, or None if the field is not a stratum."""
    if experiment_field.is_strata:
        return capi.Stratum(field_name=experiment_field.field_name)
    return None


def design_spec_filters_from_experiment(experiment: tables.Experiment) -> list[capi.Filter]:
    """Return design-spec filters from experiment_fields sorted by their position."""
    position_filters: list[tuple[int, capi.Filter]] = []
    for ef in experiment.experiment_fields:
        if api_filters := _convert_experiment_field_to_api_filters(ef):
            position_filters.extend(api_filters)

    return [f[1] for f in sorted(position_filters, key=operator.itemgetter(0))]


def design_spec_metrics_from_experiment(experiment: tables.Experiment) -> list[capi.DesignSpecMetricRequest]:
    """Return design-spec metrics from experiment_fields."""
    metrics = []
    for ef in experiment.experiment_fields:
        if api_metric := _convert_experiment_field_to_api_metric(ef):
            metrics.append(api_metric)
    return metrics


def design_spec_strata_from_experiment(experiment: tables.Experiment) -> list[capi.Stratum]:
    """Return design-spec strata from experiment_fields."""
    strata = []
    for ef in experiment.experiment_fields:
        if api_stratum := _convert_experiment_field_to_api_stratum(ef):
            strata.append(api_stratum)
    return strata


def field_type_map_from_experiment(experiment: tables.Experiment) -> dict[str, DataType]:
    """Return a map of field names used in the experiment to their data types."""
    return {ef.field_name: DataType(ef.data_type) for ef in experiment.experiment_fields}


async def design_spec_from_experiment(experiment: tables.Experiment) -> capi.DesignSpec:
    """Convert stored experiment metadata to a DesignSpec object."""
    base_experiment_dict = {
        "experiment_type": experiment.experiment_type,
        "experiment_name": experiment.name,
        "description": experiment.description,
        "design_url": experiment.design_url or None,
        "start_date": experiment.start_date,
        "end_date": experiment.end_date,
    }
    await experiment.awaitable_attrs.arms

    if experiment.experiment_type in {
        ExperimentsType.FREQ_ONLINE.value,
        ExperimentsType.FREQ_PREASSIGNED.value,
    }:
        await experiment.awaitable_attrs.experiment_fields
        primary_key_field = experiment.unique_id_field()
        if experiment.datasource_table is None or primary_key_field is None:
            raise ValueError(
                f"Frequentist experiment {experiment.id} "
                "is missing datasource_table or unique participant key field."
            )

        return TypeAdapter(capi.DesignSpec).validate_python({
            **base_experiment_dict,
            "table_name": experiment.datasource_table,
            "primary_key": primary_key_field.field_name,
            "arms": [
                {
                    "arm_id": arm.id,
                    "arm_name": arm.name,
                    "arm_description": arm.description,
                    "arm_weight": arm.arm_weight,
                }
                for arm in experiment.arms
            ],
            "strata": design_spec_strata_from_experiment(experiment),
            "metrics": design_spec_metrics_from_experiment(experiment),
            "filters": design_spec_filters_from_experiment(experiment),
            "power": experiment.power,
            "alpha": experiment.alpha,
            "fstat_thresh": experiment.fstat_thresh,
        })

    if experiment.experiment_type in {
        ExperimentsType.MAB_ONLINE.value,
        ExperimentsType.CMAB_ONLINE.value,
    }:
        if not experiment.prior_type or not experiment.reward_type:
            raise ValueError(f"Bandit experiment {experiment.id} must have prior_type and reward_type set.")
        contexts = None
        if experiment.experiment_type == ExperimentsType.CMAB_ONLINE.value:
            contexts = [
                capi.Context(
                    context_id=context.id,
                    context_name=context.name,
                    context_description=context.description,
                    value_type=capi.ContextType(context.value_type),
                )
                for context in experiment.contexts
            ]

        return TypeAdapter(capi.DesignSpec).validate_python({
            **base_experiment_dict,
            "arms": [
                {
                    "arm_id": arm.id,
                    "arm_name": arm.name,
                    "arm_description": arm.description,
                    "arm_weight": arm.arm_weight,
                    "mu_init": arm.mu_init,
                    "sigma_init": arm.sigma_init,
                    "alpha_init": arm.alpha_init,
                    "beta_init": arm.beta_init,
                    "mu": arm.mu,
                    "covariance": arm.covariance,
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                }
                for arm in experiment.arms
            ],
            "prior_type": capi.PriorTypes(experiment.prior_type),
            "reward_type": capi.LikelihoodTypes(experiment.reward_type),
            "contexts": contexts,
        })
    raise ValueError(f"Unsupported experiment type: {experiment.experiment_type}")


def balance_check_from_experiment(experiment: tables.Experiment) -> capi.BalanceCheck | None:
    if experiment.balance_check is not None:
        return capi.BalanceCheck.model_validate(experiment.balance_check)
    return None


def power_response_from_experiment(experiment: tables.Experiment) -> capi.PowerResponse | None:
    if experiment.power_analyses is None:
        return None
    return capi.PowerResponse.model_validate(experiment.power_analyses)


async def experiment_config_from_experiment(
    experiment: tables.Experiment,
    assign_summary: capi.AssignSummary,
    webhook_ids: list[str] | None = None,
) -> capi.GetExperimentResponse:
    """Construct an ExperimentConfig from the internal Experiment and an AssignSummary."""
    return capi.GetExperimentResponse(
        experiment_id=(await experiment.awaitable_attrs.id),
        datasource_id=experiment.datasource_id,
        participant_type_deprecated=experiment.participant_type,
        state=ExperimentState(experiment.state),
        stopped_assignments_at=experiment.stopped_assignments_at,
        stopped_assignments_reason=StopAssignmentReason.from_str(experiment.stopped_assignments_reason),
        design_spec=await design_spec_from_experiment(experiment),
        power_analyses=power_response_from_experiment(experiment),
        assign_summary=assign_summary,
        webhooks=webhook_ids or [],
        decision=experiment.decision,
        impact=experiment.impact,
    )


async def get_experiment_response_from_experiment(
    experiment: tables.Experiment,
    assign_summary: capi.AssignSummary,
    webhook_ids: list[str] | None = None,
) -> capi.GetExperimentResponse:
    # Although GetExperimentResponse is a subclass of ExperimentConfig, revalidate in case the API diverges.
    return capi.GetExperimentResponse.model_validate(
        (await experiment_config_from_experiment(experiment, assign_summary, webhook_ids)).model_dump()
    )


async def create_experiment_response_from_experiment(
    experiment: tables.Experiment,
    assign_summary: capi.AssignSummary,
    webhook_ids: list[str] | None = None,
) -> capi.CreateExperimentResponse:
    # Revalidate in case CreateExperimentResponse diverges from ExperimentConfig.
    return capi.CreateExperimentResponse.model_validate(
        (await experiment_config_from_experiment(experiment, assign_summary, webhook_ids)).model_dump()
    )
