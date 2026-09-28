"""Checks that every route's path parameters are consumed by the handler or its
dependencies.

Route path parameters are declared in the path template using {brackets}; they
are mapped to Python variables via method arguments. The decorator is within a
few lines of the argument's declaration, making this an intuitive and eloquent
way to declare the mappings. However, the path parameters can also be declared
on a dependency of the handler, whose implementation may be in a different
module. That causes readability issues because the declaration and usage are no
longer coincident. If the template parameters and the handler arguments don't
match, it will also be a source of bugs.

This test mitigates the risk of the reduced readability by mechanistically
verifying the path template matches the full tree of the handler's dependencies.
This detects two errors:

- An argument referring to a path parameter that doesn't exist. Requests then
  fail with a 422 naming a path parameter that cannot be supplied, and the
  OpenAPI schema advertises a parameter that isn't in the path template.
- A path parameter not consumed by an argument. E.g. a route under
  /organizations/{organization_id} whose handler takes the datasource-scoped
  dependency instead of the organization-scoped one drops the organization
  constraint without any symptoms, which would probably be a bug.

Ruff's FAST003 only does half of that: it detects mismatches between the
template and the handler arguments, but does not traverse the handler's
arguments to verify the transitive closure of dependencies.

Note this is only matching by "name" of arguments, not type -- it is always up
to the implementers of the handlers to ensure the parameters match desired
functionality. Viewing the generated OpenAPI spec or the generated API docs
is a good way to verify that the API makes sense.
"""

from collections.abc import Iterable, Iterator
from typing import Annotated

from fastapi import Depends, FastAPI, Path, params
from fastapi.dependencies.utils import get_flat_params
from fastapi.routing import APIRoute
from starlette.routing import BaseRoute, compile_path

from xngin.apiserver.main import app

# A route that only the recursive walk below can reach, so that a change to how routers are included
# fails this test rather than silently emptying it.
CANARY_ROUTE = "/v1/m/organizations/{organization_id}/datasources/{datasource_id}/experiments/{experiment_id}/snapshots"


def _api_routes(routes: Iterable[BaseRoute]) -> Iterator[APIRoute]:
    """Yields every APIRoute in the app, descending into included routers."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        elif hasattr(route, "original_router"):
            yield from _api_routes(route.original_router.routes)


def _mismatches(routes: Iterable[BaseRoute]) -> list[str]:
    """Describes every route whose path parameters disagree with what its signatures consume."""
    described = []
    for route in _api_routes(routes):
        declared = {p.alias for p in get_flat_params(route.dependant) if isinstance(p.field_info, params.Path)}
        _, _, converters = compile_path(route.path)
        in_path = set(converters)
        if declared == in_path:
            continue
        detail = []
        if unconsumed := sorted(in_path - declared):
            detail.append(f"named in the path but consumed by nothing: {unconsumed}")
        if unfillable := sorted(declared - in_path):
            detail.append(f"required by the handler or a dependency but absent from the path: {unfillable}")
        methods = "/".join(sorted(route.methods or []))
        described.append(f"{route.name} ({methods} {route.path}): {'; '.join(detail)}")
    return described


def test_path_parameters_match_what_handlers_and_dependencies_consume():
    covered = {route.path for route in _api_routes(app.routes)}
    assert CANARY_ROUTE in covered, "The route walk missed the admin router; _api_routes needs updating."

    mismatches = _mismatches(app.routes)
    assert not mismatches, "Routes whose path parameters disagree with their signatures:\n" + "\n".join(mismatches)


def test_both_kinds_of_disagreement_are_reported():
    """Guards the check itself: a passing suite never exercises the reporting above."""

    def needs_organization(organization_id: Annotated[str, Path()], experiment_id: Annotated[str, Path()]) -> str:
        return organization_id

    def ignores_organization(experiment_id: Annotated[str, Path()]) -> str:
        return experiment_id

    probe = FastAPI()

    @probe.get("/experiments/{experiment_id}")
    def wants_a_parameter_the_path_lacks(dep: Annotated[str, Depends(needs_organization)]) -> str:
        return dep

    @probe.get("/organizations/{organization_id}/experiments/{experiment_id}")
    def drops_the_organization_scope(dep: Annotated[str, Depends(ignores_organization)]) -> str:
        return dep

    reported = _mismatches(probe.routes)
    assert reported == [
        "wants_a_parameter_the_path_lacks (GET /experiments/{experiment_id}): required by the handler or a "
        "dependency but absent from the path: ['organization_id']",
        "drops_the_organization_scope (GET /organizations/{organization_id}/experiments/{experiment_id}): "
        "named in the path but consumed by nothing: ['organization_id']",
    ]
