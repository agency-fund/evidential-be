import sqlalchemy
from sqlalchemy import or_, func, ColumnOperators, Table
from sqlalchemy.orm import Session

from xngin.apiserver.api_types import AudienceSpec, Relation, AudienceSpecFilter
from xngin.apiserver.settings import ClientConfigType, get_sqlalchemy_table_from_engine


def get_metric_meta(metrics: list[str], audience_spec: AudienceSpec):
    # Implement logic to get metric metadata
    raise NotImplementedError()


def get_dwh_participants(
    config: ClientConfigType, audience_spec: AudienceSpec, chosen_n: int
):
    """get_dwh_participants resembles the dwh.R implementation.

    Not ported:
    * type_map support (why: told this isn't used anymore)
    * _latest hack (why: not sure what it is)
    * unpacking experiment_ids into a list
    """
    unit_type = audience_spec.type
    with config.dbsession(unit_type) as session:
        sa_table = get_sqlalchemy_table_from_engine(session.get_bind(), unit_type)
        # TODO: sheetconfig contains the assumptions that the experiment designers have
        # made about the data warehouse table. We should compare that data against the
        # actual schema to ensure that the comparators will behave as expected.
        filters = create_filters(sa_table, audience_spec)
        query = compose_query(session, sa_table, chosen_n, filters)
        return query.all()


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
