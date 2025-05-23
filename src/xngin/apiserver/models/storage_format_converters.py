from pydantic import TypeAdapter
from xngin.apiserver.models.tables import Experiment
from xngin.apiserver.routers.stateless_api_types import DesignSpec
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


class DesignSpecStorageConverter:
    """Converts a DesignSpec to storage components and vice versa."""

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
        # The `value` field in StorageFilter is Sequence[Any].
        # Pydantic will validate when creating ApiFilter.
        return [
            ApiFilter(
                field_name=f.field_name,
                relation=f.relation,
                value=f.value,
            )
            for f in design_spec_fields.filters
        ]

    @staticmethod
    def to_store_fields(design_spec: DesignSpec) -> DesignSpecFields:
        """Converts a DesignSpec to a DesignSpecFields object."""
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

        return DesignSpecFields(
            strata=storage_strata,
            metrics=storage_metrics,
            filters=storage_filters,
        )

    @staticmethod
    def get_api_design_spec(experiment: Experiment) -> DesignSpec:
        """Converts a DesignSpecFields to a DesignSpec object."""
        design_spec_fields = DesignSpecFields.model_validate(
            experiment.design_spec_fields
        )
        return TypeAdapter(DesignSpec).validate_python({
            "participant_type": experiment.participant_type,
            "experiment_id": experiment.id,
            "experiment_type": experiment.experiment_type,
            "experiment_name": experiment.name,
            "description": experiment.description,
            "start_date": experiment.start_date,
            "end_date": experiment.end_date,
            "arms": [
                {
                    "arm_id": arm.id,
                    "arm_name": arm.name,
                    "arm_description": arm.description,
                }
                for arm in experiment.arms
            ],
            "strata": DesignSpecStorageConverter.get_api_strata(design_spec_fields),
            "metrics": DesignSpecStorageConverter.get_api_metrics(design_spec_fields),
            "filters": DesignSpecStorageConverter.get_api_filters(design_spec_fields),
            "power": experiment.power,
            "alpha": experiment.alpha,
            "fstat_thresh": experiment.fstat_thresh,
        })
