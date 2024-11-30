import re

import sqlalchemy
from pydantic import BaseModel
from sqlalchemy import or_, func, ColumnOperators, Table, not_, select
from sqlalchemy.orm import Session

from xngin.apiserver.api_types import (
    AudienceSpec,
    Relation,
    AudienceSpecFilter,
    EXPERIMENT_IDS_SUFFIX,
    Stats,
    MetricType,
)
from xngin.apiserver.settings import ClientConfigType, get_sqlalchemy_table_from_engine
from xngin.sqlite_extensions import custom_functions


def get_metric_meta():
    # TODO: implement
    pass


class MetricStats(BaseModel):
    metric: str
    metric_type: MetricType
    stats: Stats


def get_stats_on_metrics(
    session, sa_table: Table, metric_names: list[str], audience_spec: AudienceSpec
) -> list[MetricStats]:
    metric_columns = []
    metric_names = sorted(metric_names)
    for metric in metric_names:
        col = sa_table.c[metric]
        # TODO(roboton): consider whether mitigations for null are important
        metric_columns.extend((
            func.avg(col).label(f"{metric}__mean"),
            custom_functions.stddev_pop(col).label(f"{metric}__sd"),
            func.count(col).label(f"{metric}__count"),
        ))
    query = select(*metric_columns)
    filters = create_filters(sa_table, audience_spec)
    query = query.filter(*filters)
    stats = session.execute(query).mappings().fetchone()
    return [
        MetricStats(
            metric=metric,
            metric_type=MetricType.from_python_type(
                sa_table.c[metric].type.python_type
            ),
            stats=Stats(
                mean=stats[f"{metric}__mean"],
                stddev=stats[f"{metric}__sd"],
                available_n=stats[f"{metric}__count"],
            ),
        )
        for metric in metric_names
    ]


def get_dwh_participants(
    config: ClientConfigType, audience_spec: AudienceSpec, chosen_n: int
):
    """get_dwh_participants resembles the dwh.R implementation."""
    participant_type = audience_spec.participant_type
    with config.dbsession(participant_type) as session:
        sa_table = get_sqlalchemy_table_from_engine(
            session.get_bind(), participant_type
        )
        # TODO: sheetconfig contains the assumptions that the experiment designers have
        # made about the data warehouse table. We should compare that data against the
        # actual schema to ensure that the comparators will behave as expected.
        filters = create_filters(sa_table, audience_spec)
        query = compose_query(session, sa_table, chosen_n, filters)
        return query.all()


# TODO: rename for clarity
def create_filters(sa_table: sqlalchemy.Table, audience_spec: AudienceSpec):
    """Converts an AudienceSpec into a list of SQLAlchemy filters."""

    def create_one_filter(filter_, sa_table):
        if filter_.filter_name.endswith(EXPERIMENT_IDS_SUFFIX):
            return create_special_experiment_id_filter(
                sa_table.columns[filter_.filter_name], filter_
            )
        return create_filter(sa_table.columns[filter_.filter_name], filter_)

    return [create_one_filter(filter_, sa_table) for filter_ in audience_spec.filters]


def create_special_experiment_id_filter(
    col: sqlalchemy.Column, filter_: AudienceSpecFilter
) -> ColumnOperators:
    matching_regex = make_csv_regex(filter_.value)
    match filter_.relation:
        case Relation.INCLUDES:
            return func.lower(col).regexp_match(matching_regex)
        case Relation.EXCLUDES:
            return or_(
                col.is_(None),
                func.char_length(col) == 0,
                not_(func.lower(col).regexp_match(matching_regex)),
            )


def make_csv_regex(values):
    """Constructs a regular expression for matching a CSV string against a list of values.

    The generated regexp is to be used by re.search() or equivalent. We assume that most database engines
    will support identical syntax.
    """
    value_regexp = (
        r"("
        + r"|".join(re.escape(str(v).lower()) for v in values if v is not None)
        + r")"
    )
    return r"(^x$)|(^x,)|(,x$)|(,x,)".replace("x", value_regexp)


def create_filter(
    col: sqlalchemy.Column, filter_: AudienceSpecFilter
) -> ColumnOperators:
    """Converts a single AudienceSpecFilter to a sqlalchemy filter."""
    match filter_.relation:
        case Relation.BETWEEN:
            match filter_.value:
                case (left, None):
                    return col >= left
                case (None, right):
                    return col <= right
                case (left, right):
                    return col.between(left, right)
        case Relation.EXCLUDES:
            return or_(col.is_(None), col.not_in(filter_.value))
        case Relation.INCLUDES:
            return col.in_(filter_.value)


def compose_query(session: Session, sa_table: Table, chosen_n: int, filters):
    query = session.query(sa_table)
    filtered = query.filter(*filters)
    ordered = filtered.order_by(custom_functions.our_random(sa_table))
    return ordered.limit(chosen_n)
