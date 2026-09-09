"""Eager-loading options shared by the routers' resource dependencies.

Routers describe what a route needs eagerly loaded as two lists: `preload` for relationships hanging
directly off the queried entity, and `nested_preload` for chains that walk through one relationship to
reach another. build_preload_options turns both into SQLAlchemy loader options.
"""

import enum

from sqlalchemy.orm import QueryableAttribute, joinedload, selectinload
from sqlalchemy.sql.base import ExecutableOption

from xngin.apiserver.sqla import tables


class PreloadMethod(enum.Enum):
    """Methods we'll support for preloading a SQLAlchemy relationship."""

    SELECTINLOAD = enum.auto()
    JOINLOAD = enum.auto()


type PreloadChain = list[tuple[PreloadMethod, QueryableAttribute]]

EXPERIMENT_FIELDS_WITH_FILTERS: PreloadChain = [
    (PreloadMethod.SELECTINLOAD, tables.Experiment.experiment_fields),
    (PreloadMethod.JOINLOAD, tables.ExperimentField.experiment_filters),
]


def build_preload_options(
    preload: list[QueryableAttribute] | None = None,
    nested_preload: list[PreloadChain] | None = None,
) -> list[ExecutableOption]:
    """Returns the SQLAlchemy query options for the given direct and nested relationships."""
    options: list[ExecutableOption] = []
    if preload:
        options.extend(selectinload(f) for f in preload)
    for chain in nested_preload or []:
        nested_load = None
        for method, attr in chain:
            match method:
                case PreloadMethod.SELECTINLOAD:
                    nested_load = selectinload(attr) if nested_load is None else nested_load.selectinload(attr)
                case PreloadMethod.JOINLOAD:
                    nested_load = joinedload(attr) if nested_load is None else nested_load.joinedload(attr)
        if nested_load is not None:
            options.append(nested_load)
    return options
