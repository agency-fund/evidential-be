import sqlalchemy
from sqlalchemy import or_, func, ColumnOperators, Table, select
from sqlalchemy.orm import Session

from xngin.apiserver.api_types import AudienceSpec, Relation, AudienceSpecFilter
from xngin.apiserver.settings import ClientConfigType, get_sqlalchemy_table_from_engine
# from xngin.sqlite_extensions import custom_functions


def get_metric_meta():
    # TODO: implement
    pass


def get_metric_meta_column_stats(
    session, sa_table: Table, metric_names: list[str], audience_spec: AudienceSpec
):
    """TODO: WIP for column metadata"""
    metric_columns = []
    for metric in sorted(metric_names):
        col = sa_table.c[metric]
        # Note: R code used . as separator; that causes pain with JSON and would require quoting in SQL so we use
        # double underscores instead.
        mean = func.avg(col).label(f"{metric}__metric_mean")
        stddev = func.stddev(col).label(f"{metric}__metric_sd")
        available_n = func.count(col).label(f"{metric}__metric_count")
        metric_columns.extend((mean, stddev, available_n))
    query = select(*metric_columns)
    filters = create_filters(sa_table, audience_spec)
    query = query.filter(*filters)
    return session.execute(query).fetchone()._mapping  # hack


def get_dwh_participants(
    config: ClientConfigType, audience_spec: AudienceSpec, chosen_n: int
):
    """get_dwh_participants resembles the dwh.R implementation.

    Not ported:
    * type_map support (why: told this isn't used anymore)
    * _latest hack (why: not sure what it is)
    * unpacking experiment_ids into a list
    """
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
    return [
        create_filter(sa_table.columns[filter_.filter_name], filter_)
        for filter_ in audience_spec.filters
    ]


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
    # TODO: func.random works only with postgres and sqlite (not mysql)
    return (
        session.query(sa_table).filter(*filters).order_by(func.random()).limit(chosen_n)
    )
