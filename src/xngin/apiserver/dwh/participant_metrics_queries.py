from sqlalchemy import Float, Integer, Label, String, Table, cast, select
from sqlalchemy.orm import Session

from xngin.apiserver.dwh.analysis_types import MetricValue, ParticipantOutcome
from xngin.apiserver.dwh.query_constructors import create_one_filter
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.routers.common_api_types import DesignSpecMetricRequest, Filter
from xngin.apiserver.routers.common_enums import MetricType, Relation


def get_participant_metrics(
    session: Session,
    sa_table: Table,
    metrics: list[DesignSpecMetricRequest],
    unique_id_field: str,
    participant_ids: list[str],
) -> list[ParticipantOutcome]:
    missing_metrics = {m.field_name for m in metrics if m.field_name not in sa_table.c}
    if len(missing_metrics) > 0:
        raise LateValidationError(f"Missing metrics (check your Datasource configuration): {missing_metrics}")

    metric_types = [MetricType.from_python_type(sa_table.c[m.field_name].type.python_type) for m in metrics]
    if unique_id_field not in sa_table.columns:
        raise LateValidationError(f"Unique ID field {unique_id_field} not found in table.")
    participant_id_column = sa_table.c[unique_id_field]
    select_columns: list[Label] = [cast(participant_id_column, String).label("participant_id")]

    field_names = ["participant_id"]
    for metric, metric_type in zip(metrics, metric_types, strict=False):
        field_name = metric.field_name
        field_names.append(field_name)
        col = sa_table.c[field_name]
        if metric_type is MetricType.NUMERIC:
            cast_column = cast(col, Float)
        else:
            cast_column = cast(cast(col, Integer), Float)
        select_columns.append(cast_column.label(field_name))

    participant_id_filter = Filter(
        field_name=unique_id_field,
        relation=Relation.INCLUDES,
        value=[Filter.cast_participant_id(pid, participant_id_column.type) for pid in participant_ids],
    )
    participant_filter = create_one_filter(participant_id_filter, sa_table)
    results = session.execute(select(*select_columns).filter(participant_filter))

    participant_outcomes: list[ParticipantOutcome] = []
    for result in results:
        metric_values: list[MetricValue] = []
        participant_id = None
        for i, field_name in enumerate(field_names):
            if field_name == "participant_id":
                participant_id = result[i]
            else:
                metric_values.append(MetricValue(metric_name=field_name, metric_value=result[i]))
        if participant_id is None:
            raise LateValidationError("Participant ID is required.")
        participant_outcomes.append(ParticipantOutcome(participant_id=str(participant_id), metric_values=metric_values))
    return participant_outcomes
