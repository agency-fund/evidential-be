"""Converts between API and jsonb storage models used by our internal database models.

Use these converters to set/get the different JSONB columns of their respective SQLAlchemy models.
"""

from typing import Self
from pydantic import TypeAdapter
from xngin.apiserver.models.enums import ExperimentState
from xngin.apiserver.models import tables
from xngin.apiserver.routers.experiments_api_types import (
    AssignSummary,
    CreateExperimentResponse,
    ExperimentConfig,
    GetExperimentResponse,
)
from xngin.apiserver.routers.stateless_api_types import (
    BalanceCheck,
    DesignSpec,
    PowerResponse,
    Relation,
)
from xngin.apiserver.routers.stateless_api_types import (
    Stratum as ApiStratum,
    DesignSpecMetricRequest as ApiDesignSpecMetricRequest,
    Filter as ApiFilter,
)
from xngin.apiserver.models.storage_types import (
    DesignSpecFields,
    StorageFilter,
    StorageMetric,
    StorageStratum,
)


class ExperimentStorageConverter:
    """Converts API components to storage components and vice versa for an Experiment."""

    def __init__(self, experiment: tables.Experiment):
        self.experiment = experiment

    def get_experiment(self) -> tables.Experiment:
        return self.experiment

    @staticmethod
    def get_api_strata(design_spec_fields: DesignSpecFields) -> list[ApiStratum]:
        """Converts stored strata to API Stratum objects."""
        if design_spec_fields.strata is None:
            return []
        return [ApiStratum(field_name=s.field_name) for s in design_spec_fields.strata]

    @staticmethod
    def get_api_metrics(
        design_spec_fields: DesignSpecFields,
    ) -> list[ApiDesignSpecMetricRequest]:
        """Converts stored metrics to API DesignSpecMetricRequest objects."""
        if design_spec_fields.metrics is None:
            return []
        return [
            ApiDesignSpecMetricRequest(
                field_name=m.field_name,
                metric_pct_change=m.metric_pct_change,
                metric_target=m.metric_target,
            )
            for m in design_spec_fields.metrics
        ]

    @staticmethod
    def get_api_filters(design_spec_fields: DesignSpecFields) -> list[ApiFilter]:
        """Converts stored filters to API Filter objects."""
        if design_spec_fields.filters is None:
            return []
        return [
            # The `value` field in StorageFilter is Sequence[Any].
            # Pydantic will validate when creating ApiFilter.
            ApiFilter(
                field_name=f.field_name,
                relation=Relation(f.relation),
                value=f.value,
            )
            for f in design_spec_fields.filters
        ]

    def set_design_spec_fields(self, design_spec: DesignSpec) -> Self:
        """Saves the components of a DesignSpec to the experiment."""
        storage_strata = None
        if design_spec.strata:
            storage_strata = [
                StorageStratum(field_name=s.field_name) for s in design_spec.strata
            ]

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
        return DesignSpecFields.model_validate(self.experiment.design_spec_fields)

    def get_design_spec(self) -> DesignSpec:
        """Converts a DesignSpecFields to a DesignSpec object."""
        design_spec_fields = self.get_design_spec_fields()
        return TypeAdapter(DesignSpec).validate_python({
            "participant_type": self.experiment.participant_type,
            "experiment_id": self.experiment.id,
            "experiment_type": self.experiment.experiment_type,
            "experiment_name": self.experiment.name,
            "description": self.experiment.description,
            "start_date": self.experiment.start_date,
            "end_date": self.experiment.end_date,
            "arms": [
                {
                    "arm_id": arm.id,
                    "arm_name": arm.name,
                    "arm_description": arm.description,
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

    def set_balance_check(self, value: BalanceCheck | None) -> Self:
        if value is None:
            self.experiment.balance_check = None
        else:
            self.experiment.balance_check = BalanceCheck.model_validate(
                value
            ).model_dump()
        return self

    def get_balance_check(self) -> BalanceCheck | None:
        if self.experiment.balance_check is not None:
            return BalanceCheck.model_validate(self.experiment.balance_check)
        return None

    def set_power_response(self, value: PowerResponse | None) -> Self:
        if value is None:
            self.experiment.power_analyses = None
        else:
            self.experiment.power_analyses = PowerResponse.model_validate(
                value
            ).model_dump()
        return self

    def get_power_response(self) -> PowerResponse | None:
        if self.experiment.power_analyses is None:
            return None
        return PowerResponse.model_validate(self.experiment.power_analyses)

    def get_experiment_config(self, assign_summary: AssignSummary) -> ExperimentConfig:
        """Construct an ExperimentConfig from the internal Experiment and an AssignSummary.

        Expects assign_summary since that typically requires a db lookup."""
        return ExperimentConfig(
            datasource_id=self.experiment.datasource_id,
            state=ExperimentState(self.experiment.state),
            design_spec=self.get_design_spec(),
            power_analyses=self.get_power_response(),
            assign_summary=assign_summary,
        )

    def get_experiment_response(
        self, assign_summary: AssignSummary
    ) -> GetExperimentResponse:
        # Although ListExperimentsResponse is a subclass of ExperimentConfig, we revalidate the
        # response in case we ever change the API.
        return GetExperimentResponse.model_validate(
            self.get_experiment_config(assign_summary).model_dump()
        )

    def get_create_experiment_response(
        self, assign_summary: AssignSummary
    ) -> CreateExperimentResponse:
        # Revalidate the response in case we ever change the API.
        return CreateExperimentResponse.model_validate(
            self.get_experiment_config(assign_summary).model_dump()
        )
