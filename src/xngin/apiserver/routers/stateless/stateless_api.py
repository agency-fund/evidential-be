from contextlib import asynccontextmanager
from typing import Annotated

import sqlalchemy
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Query,
    Response,
)
from loguru import logger
from sqlalchemy import distinct
from sqlalchemy.orm import Session

from xngin.apiserver import constants
from xngin.apiserver.dependencies import (
    datasource_config_required,
    gsheet_cache,
)
from xngin.apiserver.dwh.queries import get_stats_on_metrics, query_for_participants
from xngin.apiserver.dwh.reflect_schemas import create_schema_from_table
from xngin.apiserver.exceptions_common import LateValidationError
from xngin.apiserver.gsheet_cache import GSheetCache
from xngin.apiserver.routers.common_api_types import (
    DesignSpec,
    FilterClass,
    GetFiltersResponseDiscrete,
    GetFiltersResponseElement,
    GetFiltersResponseNumericOrDate,
    GetMetricsResponseElement,
    GetStrataResponseElement,
    PowerRequest,
    PowerResponse,
)
from xngin.apiserver.routers.stateless.stateless_api_types import (
    AssignRequest,
    AssignResponse,
    GetFiltersResponse,
    GetMetricsResponse,
    GetStrataResponse,
)
from xngin.apiserver.settings import (
    DatasourceConfig,
    ParticipantsConfig,
    ParticipantsMixin,
    infer_table,
)
from xngin.schema.schema_types import FieldDescriptor, ParticipantsSchema
from xngin.sheets.config_sheet import fetch_and_parse_sheet
from xngin.stats.assignment import assign_treatment as assign_treatment_actual
from xngin.stats.power import check_power


# TODO: move into its own module re: https://github.com/agency-fund/xngin/pull/188/
@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(f"Starting router: {__name__} (prefix={router.prefix})")
    yield


router = APIRouter(
    lifespan=lifespan,
    prefix=constants.API_PREFIX_V1,
)


# ruff: noqa: B903
class CommonQueryParams:
    """Describes query parameters common to the /strata, /filters, and /metrics APIs."""

    def __init__(
        self,
        participant_type: Annotated[
            str,
            Query(
                description="Unit of analysis for experiment.",
                examples=["test_participant_type"],
            ),
        ],
        refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
    ):
        self.participant_type = participant_type
        self.refresh = refresh


async def get_participants_config_and_schema(
    commons: CommonQueryParams,
    datasource_config: ParticipantsMixin,
    gsheets: GSheetCache,
) -> tuple[ParticipantsConfig, ParticipantsSchema]:
    """Get common configuration info for various endpoints."""
    participants_cfg = datasource_config.find_participants(commons.participant_type)
    cached_schema = participants_cfg  # assume type == "schema"
    if participants_cfg.type == "sheet":
        sheet_ref = participants_cfg.sheet
        cached_schema = await gsheets.get(
            sheet_ref,
            lambda: fetch_and_parse_sheet(sheet_ref),
            refresh=commons.refresh,
        )
    return participants_cfg, cached_schema


# API Endpoints
@router.get(
    "/strata",
    summary="Get possible strata covariates.",
)
async def get_strata(
    commons: Annotated[CommonQueryParams, Depends()],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    config: Annotated[DatasourceConfig, Depends(datasource_config_required)],
) -> GetStrataResponse:
    """Get possible strata covariates for a given participant type."""
    participants_cfg, schema = await get_participants_config_and_schema(
        commons, config, gsheets
    )
    strata_fields = {c.field_name: c for c in schema.fields if c.is_strata}

    with config.dbsession() as dwh_session:
        sa_table = infer_table(
            dwh_session.get_bind(),
            participants_cfg.table_name,
            config.supports_reflection(),
        )
        db_schema = generate_field_descriptors(sa_table, schema.get_unique_id_field())

    return GetStrataResponse(
        results=sorted(
            [
                GetStrataResponseElement(
                    data_type=db_schema.get(field_name).data_type,
                    field_name=field_name,
                    description=field_descriptor.description,
                    # For strata columns, we will echo back any extra annotations
                    extra=field_descriptor.extra,
                )
                for field_name, field_descriptor in strata_fields.items()
                if db_schema.get(field_name)
            ],
            key=lambda item: item.field_name,
        )
    )


@router.get(
    "/filters", summary="Get possible filters covariates for a given participant type."
)
async def get_filters(
    commons: Annotated[CommonQueryParams, Depends()],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    config: Annotated[DatasourceConfig, Depends(datasource_config_required)],
) -> GetFiltersResponse:
    participants_cfg, schema = await get_participants_config_and_schema(
        commons, config, gsheets
    )
    filter_fields = {c.field_name: c for c in schema.fields if c.is_filter}

    with config.dbsession() as dwh_session:
        sa_table = infer_table(
            dwh_session.get_bind(),
            participants_cfg.table_name,
            config.supports_reflection(),
        )
        db_schema = generate_field_descriptors(sa_table, schema.get_unique_id_field())

        mapper = create_col_to_filter_meta_mapper(db_schema, sa_table, dwh_session)

        return GetFiltersResponse(
            results=sorted(
                [
                    mapper(field_name, field_descriptor)
                    for field_name, field_descriptor in filter_fields.items()
                    if db_schema.get(field_name)
                ],
                key=lambda item: item.field_name,
            )
        )


def create_col_to_filter_meta_mapper(
    db_schema: dict[str, FieldDescriptor], sa_table, session: Session
):
    # TODO: implement caching, respecting commons.refresh
    def mapper(col_name, column_descriptor) -> GetFiltersResponseElement:
        db_col = db_schema.get(col_name)
        filter_class = db_col.data_type.filter_class(col_name)

        # Collect metadata on the values in the database.
        sa_col = sa_table.columns[col_name]
        match filter_class:
            case FilterClass.DISCRETE:
                distinct_values = [
                    str(v)
                    for v in session.execute(
                        sqlalchemy.select(distinct(sa_col))
                        .where(sa_col.is_not(None))
                        .limit(1000)
                        .order_by(sa_col)
                    ).scalars()
                ]
                return GetFiltersResponseDiscrete(
                    field_name=col_name,
                    data_type=db_col.data_type,
                    relations=filter_class.valid_relations(),
                    description=column_descriptor.description,
                    distinct_values=distinct_values,
                )
            case FilterClass.NUMERIC:
                min_, max_ = session.execute(
                    sqlalchemy.select(
                        sqlalchemy.func.min(sa_col), sqlalchemy.func.max(sa_col)
                    ).where(sa_col.is_not(None))
                ).first()
                return GetFiltersResponseNumericOrDate(
                    field_name=col_name,
                    data_type=db_col.data_type,
                    relations=filter_class.valid_relations(),
                    description=column_descriptor.description,
                    min=min_,
                    max=max_,
                )
            case _:
                raise RuntimeError("unexpected filter class")

    return mapper


@router.get(
    "/metrics", summary="Get possible metric covariates for a given participant type."
)
async def get_metrics(
    commons: Annotated[CommonQueryParams, Depends()],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    config: Annotated[DatasourceConfig, Depends(datasource_config_required)],
) -> GetMetricsResponse:
    """Get possible metrics for a given participant type."""
    participants_cfg, schema = await get_participants_config_and_schema(
        commons, config, gsheets
    )
    metric_cols = {c.field_name: c for c in schema.fields if c.is_metric}

    with config.dbsession() as dwh_session:
        sa_table = infer_table(
            dwh_session.get_bind(),
            participants_cfg.table_name,
            config.supports_reflection(),
        )
        db_schema = generate_field_descriptors(sa_table, schema.get_unique_id_field())

    # Merge data type info above with the columns to be used as metrics:
    return GetMetricsResponse(
        results=sorted(
            [
                GetMetricsResponseElement(
                    data_type=db_schema.get(col_name).data_type,
                    field_name=col_name,
                    description=col_descriptor.description,
                )
                for col_name, col_descriptor in metric_cols.items()
                if db_schema.get(col_name)
            ],
            key=lambda item: item.field_name,
        )
    )


def validate_schema_metrics_or_raise(
    design_spec: DesignSpec, schema: ParticipantsSchema
):
    metric_fields = {m.field_name for m in schema.fields if m.is_metric}
    metrics_requested = {m.field_name for m in design_spec.metrics}
    invalid_metrics = metrics_requested - metric_fields
    if len(invalid_metrics) > 0:
        raise LateValidationError(
            f"Invalid DesignSpec metrics (check your Datasource configuration): {invalid_metrics}"
        )


@router.post(
    "/power", summary="Check power given an experiment and audience specification."
)
async def powercheck(
    body: PowerRequest,
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    config: Annotated[DatasourceConfig, Depends(datasource_config_required)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
) -> PowerResponse:
    """Calculates statistical power given the PowerRequest details."""
    commons = CommonQueryParams(
        participant_type=body.design_spec.participant_type, refresh=refresh
    )
    participants_cfg, schema = await get_participants_config_and_schema(
        commons, config, gsheets
    )
    validate_schema_metrics_or_raise(body.design_spec, schema)

    return power_check_impl(body, config, participants_cfg)


def power_check_impl(
    body: PowerRequest, config: DatasourceConfig, participants_cfg: ParticipantsConfig
) -> PowerResponse:
    with config.dbsession() as dwh_session:
        sa_table = infer_table(
            dwh_session.get_bind(),
            participants_cfg.table_name,
            config.supports_reflection(),
        )

        metric_stats = get_stats_on_metrics(
            dwh_session,
            sa_table,
            body.design_spec.metrics,
            body.design_spec.filters,
        )

        return PowerResponse(
            analyses=check_power(
                metrics=metric_stats,
                n_arms=len(body.design_spec.arms),
                power=body.design_spec.power,
                alpha=body.design_spec.alpha,
            )
        )


@router.post(
    "/assign", summary="Assign treatment given experiment and audience specification."
)
async def assign_treatment(
    body: AssignRequest,
    chosen_n: Annotated[
        int, Query(..., description="Number of participants to assign.")
    ],
    gsheets: Annotated[GSheetCache, Depends(gsheet_cache)],
    config: Annotated[DatasourceConfig, Depends(datasource_config_required)],
    refresh: Annotated[bool, Query(description="Refresh the cache.")] = False,
    quantiles: Annotated[
        int,
        Query(
            description="Number of quantile buckets to use for stratification of numerics."
        ),
    ] = 4,
    stratum_id_name: Annotated[
        str | None,
        Query(
            description="If you wish to also retain the stratum group id per participant, provide a non-null name to output this value as an extra Strata field.",
        ),
    ] = None,
    random_state: Annotated[
        int | None,
        Query(
            description="Specify a random seed for reproducibility.",
            include_in_schema=False,
        ),
    ] = None,
) -> AssignResponse:
    commons = CommonQueryParams(
        participant_type=body.design_spec.participant_type, refresh=refresh
    )
    participants_cfg, schema = await get_participants_config_and_schema(
        commons, config, gsheets
    )
    validate_schema_metrics_or_raise(body.design_spec, schema)

    with config.dbsession() as dwh_session:
        sa_table = infer_table(
            dwh_session.get_bind(),
            participants_cfg.table_name,
            config.supports_reflection(),
        )
        participants = query_for_participants(
            dwh_session, sa_table, body.design_spec.filters, chosen_n
        )

    metric_names = [m.field_name for m in body.design_spec.metrics]
    strata_names = [s.field_name for s in body.design_spec.strata]
    return assign_treatment_actual(
        sa_table=sa_table,
        data=participants,
        stratum_cols=strata_names + metric_names,
        id_col=schema.get_unique_id_field(),
        arms=body.design_spec.arms,
        experiment_id=body.design_spec.experiment_id,
        fstat_thresh=body.design_spec.fstat_thresh,
        quantiles=quantiles,
        stratum_id_name=stratum_id_name,
        random_state=random_state,
    )


@router.get("/_authcheck", include_in_schema=False, status_code=204)
def authcheck(
    _config: Annotated[DatasourceConfig, Depends(datasource_config_required)],
):
    """Returns 204 if the request is allowed to use the requested datasource."""
    return Response(status_code=204)


def generate_field_descriptors(table: sqlalchemy.Table, unique_id_col: str):
    """Fetches a map of column name to schema metadata.

    Uniqueness of the values in the column unique_id_col is assumed, not verified!
    """
    return {
        c.field_name: c for c in create_schema_from_table(table, unique_id_col).fields
    }
