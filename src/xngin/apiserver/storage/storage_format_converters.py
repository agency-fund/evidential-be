"""Converts between API and jsonb storage models used by our internal database models.

To better decouple the API from the storage models, this file defines helpers to set/get JSONB
columns of their respective SQLALchemy models, and construct API types from SQLA/jsonb storage
types. Our SQLA tables ideally shouldn't depend on xngin/apiserver/*; but for those that declare
JSONB type columns for multi-value/complex types, use the converters to get/set them properly.
"""

from datetime import datetime
from typing import Self

import numpy as np
from pydantic import TypeAdapter

from xngin.apiserver.routers import common_api_types as capi
from xngin.apiserver.routers.common_enums import (
    ExperimentState,
    ExperimentsType,
    StopAssignmentReason,
)
from xngin.apiserver.sqla import tables
from xngin.apiserver.storage.storage_types import (
    DesignSpecFields,
    StorageFilter,
    StorageMetric,
    StorageStratum,
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

    @staticmethod
    def get_api_strata(
        design_spec_fields: DesignSpecFields,
    ) -> list[capi.Stratum]:
        """Converts stored strata to API Stratum objects."""
        if design_spec_fields.strata is None:
            return []
        return [capi.Stratum(field_name=s.field_name) for s in design_spec_fields.strata]

    @staticmethod
    def get_api_metrics(
        design_spec_fields: DesignSpecFields,
    ) -> list[capi.DesignSpecMetricRequest]:
        """Converts stored metrics to API DesignSpecMetricRequest objects."""
        if design_spec_fields.metrics is None:
            return []
        return [
            capi.DesignSpecMetricRequest(
                field_name=m.field_name,
                metric_pct_change=m.metric_pct_change,
                metric_target=m.metric_target,
            )
            for m in design_spec_fields.metrics
        ]

    @staticmethod
    def get_api_filters(
        design_spec_fields: DesignSpecFields,
    ) -> list[capi.Filter]:
        """Converts stored filters to API Filter objects."""
        if design_spec_fields.filters is None:
            return []
        return [
            # The `value` field in StorageFilter is Sequence[Any].
            # Pydantic will validate when creating Filter.
            capi.Filter(
                field_name=f.field_name,
                relation=capi.Relation(f.relation),
                value=f.value,
            )
            for f in design_spec_fields.filters
        ]

    def set_design_spec_fields(self, design_spec: capi.DesignSpec) -> Self:
        """Saves the components of a DesignSpec to the experiment."""
        if not isinstance(design_spec, capi.BaseFrequentistDesignSpec):
            self.experiment.design_spec_fields = None
            self.experiment.design_fields = []
            return self

        # Clear existing design fields
        self.experiment.design_fields = []

        # Add filters
        if design_spec.filters:
            for filter_item in design_spec.filters:
                self.experiment.design_fields.append(
                    tables.ExperimentField(
                        field_name=filter_item.field_name,
                        use="filter",
                        data_type=None,  # Will be populated during migration or by future logic
                        other={"relation": filter_item.relation, "value": list(filter_item.value)},
                    )
                )

        # Add metrics
        if design_spec.metrics:
            for metric in design_spec.metrics:
                other_data = {}
                if metric.metric_pct_change is not None:
                    other_data["metric_pct_change"] = metric.metric_pct_change
                if metric.metric_target is not None:
                    other_data["metric_target"] = metric.metric_target
                self.experiment.design_fields.append(
                    tables.ExperimentField(
                        field_name=metric.field_name,
                        use="metric",
                        data_type=None,  # Will be populated during migration or by future logic
                        other=other_data or None,
                    )
                )

        # Add strata
        if design_spec.strata:
            for stratum in design_spec.strata:
                self.experiment.design_fields.append(
                    tables.ExperimentField(
                        field_name=stratum.field_name,
                        use="stratum",
                        data_type=None,  # Will be populated during migration or by future logic
                        other=None,
                    )
                )

        # Keep JSONB column in sync for backwards compatibility during transition
        storage_strata = None
        if design_spec.strata:
            storage_strata = [StorageStratum(field_name=s.field_name) for s in design_spec.strata]

        storage_metrics = None
        if design_spec.metrics:
            storage_metrics = [
                StorageMetric(
                    field_name=m.field_name,
                    metric_pct_change=m.metric_pct_change,
                    metric_target=m.metric_target,
                )
                for m in design_spec.metrics
            ]

        storage_filters = None
        if design_spec.filters:
            storage_filters = [
                StorageFilter(
                    field_name=f.field_name,
                    relation=f.relation,
                    value=f.value,
                )
                for f in design_spec.filters
            ]

        self.experiment.design_spec_fields = DesignSpecFields(
            strata=storage_strata,
            metrics=storage_metrics,
            filters=storage_filters,
        ).model_dump(mode="json")
        return self

    def get_design_spec_fields(self) -> DesignSpecFields:
        """Reconstruct DesignSpecFields from design_fields relationship, which must be already eager-loaded."""
        # Fallback to JSONB column if design_fields is not loaded or empty
        # (for backwards compatibility during transition)
        if not self.experiment.design_fields:
            return DesignSpecFields.model_validate(self.experiment.design_spec_fields)

        filters = None
        metrics = None
        strata = None

        # Extract filters from design_fields
        filter_fields = [df for df in self.experiment.design_fields if df.use == "filter"]
        if filter_fields:
            filters = [
                StorageFilter(
                    field_name=df.field_name,
                    relation=df.other.get("relation") if df.other else "",
                    value=df.other.get("value") if df.other else [],
                )
                for df in filter_fields
            ]

        # Extract metrics from design_fields
        metric_fields = [df for df in self.experiment.design_fields if df.use == "metric"]
        if metric_fields:
            metrics = [
                StorageMetric(
                    field_name=df.field_name,
                    metric_pct_change=df.other.get("metric_pct_change") if df.other else None,
                    metric_target=df.other.get("metric_target") if df.other else None,
                )
                for df in metric_fields
            ]

        # Extract strata from design_fields
        strata_fields = [df for df in self.experiment.design_fields if df.use == "stratum"]
        if strata_fields:
            strata = [StorageStratum(field_name=df.field_name) for df in strata_fields]

        return DesignSpecFields(filters=filters, metrics=metrics, strata=strata)

    def get_design_spec_filters(self) -> list[capi.Filter]:
        return ExperimentStorageConverter.get_api_filters(self.get_design_spec_fields())

    def get_design_spec_metrics(self) -> list[capi.DesignSpecMetricRequest]:
        return ExperimentStorageConverter.get_api_metrics(self.get_design_spec_fields())

    async def get_design_spec(self) -> capi.DesignSpec:
        """Converts a DesignSpecFields to a DesignSpec object."""
        base_experiment_dict = {
            "participant_type": self.experiment.participant_type,
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
            # Load design_fields relationship if needed
            await self.experiment.awaitable_attrs.design_fields
            design_spec_fields = self.get_design_spec_fields()
            return TypeAdapter(capi.DesignSpec).validate_python({
                **base_experiment_dict,
                "arms": [
                    {
                        "arm_id": arm.id,
                        "arm_name": arm.name,
                        "arm_description": arm.description,
                        "arm_weight": arm.arm_weight,
                    }
                    for arm in self.experiment.arms
                ],
                "strata": ExperimentStorageConverter.get_api_strata(design_spec_fields),
                "metrics": ExperimentStorageConverter.get_api_metrics(design_spec_fields),
                "filters": ExperimentStorageConverter.get_api_filters(design_spec_fields),
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
    ) -> Self:
        """Init experiment with arms from components. Get the final object with get_experiment().

        Raises:
            ValueError: If the experiment_id is not set in the design_spec.
        """
        # Initialize common fields
        experiment = tables.Experiment(
            datasource_id=datasource_id,
            experiment_type=experiment_type,
            participant_type=design_spec.participant_type,
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
                    .set_design_spec_fields(design_spec)
                    .set_balance_check(balance_check)
                    .set_power_response(power_analyses)
                )

            case capi.BaseBanditExperimentSpec():
                if design_spec.experiment_type == ExperimentsType.CMAB_ONLINE and not design_spec.contexts:
                    raise ValueError("Contexts are required for CMAB experiments.")

                # Set bandit fields
                context_len = 1
                if design_spec.contexts:
                    context_len = len(design_spec.contexts)
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

                experiment.arms = [
                    tables.Arm(
                        name=arm.arm_name,
                        description=arm.arm_description,
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
