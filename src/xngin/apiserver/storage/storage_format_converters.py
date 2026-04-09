"""Converts between API and jsonb storage models used by our internal database models.

To better decouple the API from the storage models, this file defines helpers to set/get JSONB
columns of their respective SQLALchemy models, and construct API types from SQLA/jsonb storage
types. Our SQLA tables ideally shouldn't depend on xngin/apiserver/*; but for those that declare
JSONB type columns for multi-value/complex types, use the converters to get/set them properly.
"""

import operator
from datetime import datetime
from typing import Any, Self

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


class ExperimentStorageConverter:
    """Converts API components to storage components and vice versa for an Experiment."""

    def __init__(self, experiment: tables.Experiment):
        """
        Assemble a partial experiment with setters, and get the final object or derived API objects.
        """
        self.experiment = experiment

    def get_experiment(self) -> tables.Experiment:
        """When you're done assembling the experiment, use this to get the final object."""
        return self.experiment

    def set_experiment_fields(
        self,
        design_spec: capi.DesignSpec,
        field_type_map: dict[str, DataType] | None = None,
        unique_id_name: str | None = None,
    ) -> Self:
        """Saves the components of a DesignSpec to the experiment.

        Args:
            design_spec: The design specification to save.
            field_type_map: Optional field name to data type mapping.
            unique_id_name: Optional unique ID field name.
        """
        if not isinstance(design_spec, capi.BaseFrequentistDesignSpec):
            self.experiment.design_spec_fields = None
            self.experiment.experiment_fields = []
            return self

        field_type_map = field_type_map or {}

        # Clear existing design fields
        self.experiment.experiment_fields = []

        # New fields used in the experiment. Each key is a field name and maps to a ExperimentField object.
        fields_used_map: dict[str, tables.ExperimentField] = {}

        # Add unique ID
        if unique_id_name:
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
        self.experiment.experiment_fields = list(fields_used_map.values())

        return self

    def _convert_experiment_field_to_api_filters(
        self,
        experiment_field: tables.ExperimentField,
    ) -> list[tuple[int, capi.Filter]]:
        """Converts an ExperimentField to a list of (position, API filter) tuples, or empty list if not a filter."""
        filters = []
        for f in experiment_field.experiment_filters or []:
            # Convert storage type to API type
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
        self,
        experiment_field: tables.ExperimentField,
    ) -> capi.DesignSpecMetricRequest | None:
        """Converts an ExperimentField to an API metric, or None if the field is not a metric."""
        if experiment_field.is_metric:
            return capi.DesignSpecMetricRequest(
                field_name=experiment_field.field_name,
                metric_pct_change=experiment_field.metric_pct_change,
                metric_target=experiment_field.metric_target,
            )
        return None

    def _convert_experiment_field_to_api_stratum(
        self,
        experiment_field: tables.ExperimentField,
    ) -> capi.Stratum | None:
        """Converts an ExperimentField to an API stratum, or None if the field is not a stratum."""
        if experiment_field.is_strata:
            return capi.Stratum(field_name=experiment_field.field_name)
        return None

    def get_design_spec_filters(self) -> list[capi.Filter]:
        """Return design-spec filters from experiment_fields sorted by their position."""
        position_filters: list[tuple[int, capi.Filter]] = []
        for ef in self.experiment.experiment_fields:
            if api_filters := self._convert_experiment_field_to_api_filters(ef):
                position_filters.extend(api_filters)

        return [f[1] for f in sorted(position_filters, key=operator.itemgetter(0))]

    def get_design_spec_metrics(self) -> list[capi.DesignSpecMetricRequest]:
        """Return design-spec metrics from experiment_fields."""
        metrics = []
        for ef in self.experiment.experiment_fields:
            if api_metric := self._convert_experiment_field_to_api_metric(ef):
                metrics.append(api_metric)
        return metrics

    def get_design_spec_strata(self) -> list[capi.Stratum]:
        """Return design-spec strata from experiment_fields."""
        strata = []
        for ef in self.experiment.experiment_fields:
            if api_stratum := self._convert_experiment_field_to_api_stratum(ef):
                strata.append(api_stratum)
        return strata

    def get_field_type_map(self) -> dict[str, DataType]:
        """Return a map of all the field names used in the experiment to their respective data types."""
        return {ef.field_name: DataType(ef.data_type) for ef in self.experiment.experiment_fields}

    async def get_design_spec(self) -> capi.DesignSpec:
        """Converts stored experiment metadata to a DesignSpec object."""
        base_experiment_dict = {
            "experiment_type": self.experiment.experiment_type,
            "experiment_name": self.experiment.name,
            "description": self.experiment.description,
            "design_url": self.experiment.design_url or None,
            "start_date": self.experiment.start_date,
            "end_date": self.experiment.end_date,
        }
        await self.experiment.awaitable_attrs.arms

        if self.experiment.experiment_type in {
            ExperimentsType.FREQ_ONLINE.value,
            ExperimentsType.FREQ_PREASSIGNED.value,
        }:
            await self.experiment.awaitable_attrs.experiment_fields
            primary_key_field = self.experiment.unique_id_field()
            if self.experiment.datasource_table is None or primary_key_field is None:
                raise ValueError("Frequentist experiment is missing datasource_table or unique participant key field.")

            return TypeAdapter(capi.DesignSpec).validate_python({
                **base_experiment_dict,
                "table_name": self.experiment.datasource_table,
                "primary_key": primary_key_field.field_name,
                "arms": [
                    {
                        "arm_id": arm.id,
                        "arm_name": arm.name,
                        "arm_description": arm.description,
                        "arm_weight": arm.arm_weight,
                    }
                    for arm in self.experiment.arms
                ],
                "strata": self.get_design_spec_strata(),
                "metrics": self.get_design_spec_metrics(),
                "filters": self.get_design_spec_filters(),
                "power": self.experiment.power,
                "alpha": self.experiment.alpha,
                "fstat_thresh": self.experiment.fstat_thresh,
            })

        if self.experiment.experiment_type in {
            ExperimentsType.MAB_ONLINE.value,
            ExperimentsType.CMAB_ONLINE.value,
        }:
            if not self.experiment.prior_type or not self.experiment.reward_type:
                raise ValueError("Bandit experiments must have prior_type and reward_type set.")
            contexts = None
            if self.experiment.experiment_type == ExperimentsType.CMAB_ONLINE.value:
                contexts = [
                    capi.Context(
                        context_id=context.id,
                        context_name=context.name,
                        context_description=context.description,
                        value_type=capi.ContextType(context.value_type),
                    )
                    for context in self.experiment.contexts
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
                    for arm in self.experiment.arms
                ],
                "prior_type": capi.PriorTypes(self.experiment.prior_type),
                "reward_type": capi.LikelihoodTypes(self.experiment.reward_type),
                "contexts": contexts,
            })
        raise ValueError(f"Unsupported experiment type: {self.experiment.experiment_type}")

    def set_balance_check(self, value: capi.BalanceCheck | None) -> Self:
        if value is None:
            self.experiment.balance_check = None
        else:
            self.experiment.balance_check = capi.BalanceCheck.model_validate(value).model_dump()
        return self

    def get_balance_check(self) -> capi.BalanceCheck | None:
        if self.experiment.balance_check is not None:
            return capi.BalanceCheck.model_validate(self.experiment.balance_check)
        return None

    def set_power_response(self, value: capi.PowerResponse | None) -> Self:
        if value is None:
            self.experiment.power_analyses = None
        else:
            self.experiment.power_analyses = capi.PowerResponse.model_validate(value).model_dump()
        return self

    def get_power_response(
        self,
    ) -> capi.PowerResponse | None:
        if self.experiment.power_analyses is None:
            return None
        return capi.PowerResponse.model_validate(self.experiment.power_analyses)

    async def get_experiment_config(
        self,
        assign_summary: capi.AssignSummary,
        webhook_ids: list[str] | None = None,
    ) -> capi.GetExperimentResponse:
        """Construct an ExperimentConfig from the internal Experiment and an AssignSummary.

        Expects assign_summary since that typically requires a db lookup."""
        return capi.GetExperimentResponse(
            experiment_id=(await self.experiment.awaitable_attrs.id),
            datasource_id=self.experiment.datasource_id,
            participant_type_deprecated=self.experiment.participant_type,
            state=ExperimentState(self.experiment.state),
            stopped_assignments_at=self.experiment.stopped_assignments_at,
            stopped_assignments_reason=StopAssignmentReason.from_str(self.experiment.stopped_assignments_reason),
            design_spec=await self.get_design_spec(),
            power_analyses=self.get_power_response(),
            assign_summary=assign_summary,
            webhooks=webhook_ids or [],
            decision=self.experiment.decision,
            impact=self.experiment.impact,
        )

    async def get_experiment_response(
        self,
        assign_summary: capi.AssignSummary,
        webhook_ids: list[str] | None = None,
    ) -> capi.GetExperimentResponse:
        # Although GetExperimentResponse is a subclass of ExperimentConfig, we revalidate the
        # response in case we ever change the API.
        return capi.GetExperimentResponse.model_validate(
            (await self.get_experiment_config(assign_summary, webhook_ids)).model_dump()
        )

    async def get_create_experiment_response(
        self,
        assign_summary: capi.AssignSummary,
        webhook_ids: list[str] | None = None,
    ) -> capi.CreateExperimentResponse:
        # Revalidate the response in case we ever change the API.
        return capi.CreateExperimentResponse.model_validate(
            (await self.get_experiment_config(assign_summary, webhook_ids)).model_dump()
        )

    @classmethod
    def init_from_components(
        cls,
        datasource_id: str,
        organization_id: str,
        experiment_type: capi.ExperimentsType,
        design_spec: capi.DesignSpec,
        state: ExperimentState = ExperimentState.ASSIGNED,
        stopped_assignments_at: datetime | None = None,
        stopped_assignments_reason: StopAssignmentReason | str | None = None,
        balance_check: capi.BalanceCheck | None = None,
        power_analyses: capi.PowerResponse | None = None,
        n_trials: int = 0,
        decision: str = "",
        impact: str = "",
        table_name: str | None = None,
        field_type_map: dict[str, DataType] | None = None,
        unique_id_name: str | None = None,
        participant_type: str = "",
    ) -> Self:
        """Init experiment with arms from components. Get the final object with get_experiment().

        Raises:
            ValueError: If the experiment_id is not set in the design_spec.
        """
        # Initialize common fields
        experiment = tables.Experiment(
            datasource_id=datasource_id,
            experiment_type=experiment_type,
            participant_type=participant_type,
            datasource_table=table_name,
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
            case capi.BaseFrequentistDesignSpec():
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
                return (
                    cls(experiment)
                    .set_experiment_fields(design_spec, field_type_map, unique_id_name)
                    .set_balance_check(balance_check)
                    .set_power_response(power_analyses)
                )

            case capi.BaseBanditExperimentSpec():
                if design_spec.experiment_type == ExperimentsType.CMAB_ONLINE and not design_spec.contexts:
                    raise ValueError("Contexts are required for CMAB experiments.")

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

                return cls(experiment)
            case _:
                raise ValueError(f"Unsupported design_spec type: {type(design_spec)}.")
