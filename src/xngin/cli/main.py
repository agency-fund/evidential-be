"""Command line tool for various xngin-related operations."""

import asyncio
import functools
import json
import logging
import os
import shutil
import subprocess  # noqa: S404
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from email_validator import EmailNotValidError, validate_email
from rich.console import Console
from sqlalchemy import create_engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session

from xngin.cli.commands import create_testing_dwh as _create_testing_dwh_cmd
from xngin.cli.common import create_engine_and_database
from xngin.xsecrets import secretservice

CLI_DB_APPLICATION_NAME = f"cli-{os.getpid()}"

err_console = Console(stderr=True)
console = Console(stderr=False)
app = typer.Typer(help=__doc__)
snapshots_app = typer.Typer(help="Create and modify fake historical snapshots for development.")
app.add_typer(snapshots_app, name="snapshots")
_create_testing_dwh_cmd.register(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

secretservice.setup(allow_noop=True)

async_command = lambda f: functools.wraps(f)(lambda *args, **kwargs: asyncio.run(f(*args, **kwargs)))  # noqa: E731


class Base64OrJson(StrEnum):
    base64 = "base64"
    json = "json"


def parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


@app.command()
def create_apiserver_db(
    dsn: Annotated[
        str,
        typer.Option(help="The SQLAlchemy URL for the database.", envvar="DATABASE_URL"),
    ],
):
    from xngin.apiserver.sqla import tables  # noqa: PLC0415

    console.print(f"DSN: [cyan]{dsn}[/cyan]")
    engine = create_engine_and_database(make_url(dsn), connect_args={"application_name": CLI_DB_APPLICATION_NAME})
    tables.Base.metadata.create_all(bind=engine)


@app.command()
def export_json_schemas(output: Path = Path(".schemas")):
    """Generates JSON schemas for Xngin settings files."""
    from xngin.apiserver.dwh.inspection_types import ParticipantsSchema  # noqa: PLC0415
    from xngin.apiserver.settings import Datasource  # noqa: PLC0415

    if not output.exists():
        output.mkdir()
    for model in (ParticipantsSchema, Datasource):
        filename = output / (model.__name__ + ".schema.json")
        filename.write_text(json.dumps(model.model_json_schema(), indent=2, sort_keys=True))
        print(f"Wrote {filename}.")


@app.command()
def export_openapi_spec(output: Path = Path("openapi.json")):
    """Writes the OpenAPI spec to the file specified by --output."""
    from fastapi import FastAPI  # noqa: PLC0415

    import xngin.apiserver.openapi  # noqa: PLC0415

    app = FastAPI()
    from xngin.apiserver import routes  # noqa: PLC0415

    routes.register(app)
    with open(output, "w") as outf:
        json.dump(xngin.apiserver.openapi.custom_openapi(app), outf, sort_keys=True, indent=2)


@app.command()
def bigquery_dataset_set_default_expiration(
    project_id: Annotated[
        str,
        typer.Option(..., help="The Google Cloud Project ID containing the dataset."),
    ],
    dataset_id: Annotated[str, typer.Option(..., help="The dataset name.")],
    days: Annotated[
        int,
        typer.Option(
            ...,
            help="The default expiration for new tables in the dataset (in days).",
            min=0,
        ),
    ] = 1,
):
    """Sets the default TTL (in days) of tables created in a dataset.

    Does not apply to existing tables. To remove the expiration time, specify --days 0.

    This is useful in testing environments that create BigQuery tables that are of minimal use when testing completes.
    """
    new_expiration_ms = days * 24 * 60 * 60 * 1000
    from google.cloud import bigquery  # noqa: PLC0415

    client = bigquery.Client()
    dataset = client.get_dataset(f"{project_id}.{dataset_id}")
    dataset.default_table_expiration_ms = new_expiration_ms
    dataset = client.update_dataset(dataset, ["default_table_expiration_ms"])
    print(
        f"Updated dataset {dataset.project}.{dataset.dataset_id} with new default table "
        f"expiration {dataset.default_table_expiration_ms}"
    )


@app.command()
def bigquery_dataset_delete(
    project_id: Annotated[
        str,
        typer.Option(..., help="The Google Cloud Project ID containing the dataset."),
    ],
    dataset_id: Annotated[str, typer.Option(..., help="The dataset name.")],
):
    """Deletes a BigQuery dataset."""
    from google.cloud import bigquery  # noqa: PLC0415
    from google.cloud.exceptions import NotFound  # noqa: PLC0415

    client = bigquery.Client()
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.delete_dataset(dataset_ref, delete_contents=True)
    except NotFound as exc:
        print(f"Dataset {dataset_ref} does not exist.")
        raise typer.Exit(1) from exc
    else:
        print(f"Dataset {dataset_ref} has been deleted.")


@app.command()
def bigquery_table_delete(
    project_id: Annotated[
        str,
        typer.Option(..., help="The Google Cloud Project ID containing the dataset."),
    ],
    dataset_id: Annotated[str, typer.Option(..., help="The dataset name.")],
    table_id: Annotated[str, typer.Option(..., help="The table name.")],
):
    """Deletes a BigQuery table."""
    from google.cloud import bigquery  # noqa: PLC0415
    from google.cloud.exceptions import NotFound  # noqa: PLC0415

    client = bigquery.Client()
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        client.delete_table(table_ref)
    except NotFound as exc:
        print(f"Table {table_ref} does not exist.")
        raise typer.Exit(1) from exc
    else:
        print(f"Table {table_ref} has been deleted.")


def validate_arg_is_email(email: str):
    try:
        return validate_email(email, check_deliverability=False).normalized
    except EmailNotValidError as err:
        raise typer.BadParameter(str(err)) from err


@app.command()
@async_command
async def add_user(
    database_url: Annotated[
        str,
        typer.Option(
            help="The application database where the user should be added.",
            envvar="DATABASE_URL",
        ),
    ],
    email: Annotated[
        str | None,
        typer.Argument(
            help="Email address of the user to add. If not provided, will prompt interactively.",
            callback=lambda v: validate_arg_is_email(v) if v else None,
            envvar="XNGIN_ADD_USER_EMAIL",
        ),
    ],
    privileged: Annotated[
        bool,
        typer.Option(help="Whether the user should have privileged access."),
    ] = False,
    dwh: Annotated[
        str | None,
        typer.Option(
            help="The SQLAlchemy URI of a DWH to be added to the user's organization.",
            envvar="XNGIN_DEVDWH_DSN",
        ),
    ] = None,
):
    """Adds a new user to the database.

    This command connects to the specified database and adds a new user with the given email address.
    If the --privileged flag is set, the user will be granted privileged access.

    If email is not provided as an argument, the command will prompt for it interactively.

    This command is only useful for local development databases; do not use it against production databases.
    """
    from xngin.apiserver.sqla import tables  # noqa: PLC0415
    from xngin.apiserver.storage.bootstrap import create_entities_for_first_time_user  # noqa: PLC0415

    console.print(f"Using application database: [cyan]{database_url}[/cyan]")
    console.print(f"Using data warehouse: [cyan]{dwh}[/cyan]")

    console.print(f"Adding user with email: [cyan]{email}[/cyan]")
    console.print(f"Privileged access: [cyan]{privileged}[/cyan]")

    if not dwh:
        console.print(
            "\n[bold yellow]Warning: Not adding a datasource for a data warehouse "
            "because the --dwh flag was not specified or environment variable "
            "XNGIN_DEVDWH_DSN is unset.[/bold yellow]"
        )

    engine = create_async_engine(database_url, connect_args={"application_name": CLI_DB_APPLICATION_NAME})
    async with AsyncSession(engine) as session:
        try:
            user = await create_entities_for_first_time_user(
                session, tables.User(email=email, is_privileged=privileged), dwh
            )
            await session.commit()
            await session.refresh(user)
            console.print("\n[bold green]User added successfully:[/bold green]")
            console.print(f"User ID: [cyan]{user.id}[/cyan]")
            console.print(f"Email: [cyan]{user.email}[/cyan]")
            console.print(f"Privileged: [cyan]{user.is_privileged}[/cyan]")
            api_keys = {}
            for organization in await user.awaitable_attrs.organizations:
                console.print(f"Organization: [cyan]{organization.name}[/cyan] (ID: {organization.id})")
                for datasource in await organization.awaitable_attrs.datasources:
                    from xngin.apiserver import apikeys  # noqa: PLC0415

                    label, key = apikeys.make_key()
                    key_hash = apikeys.hash_key_or_raise(key)
                    api_keys[datasource.id] = key
                    (await datasource.awaitable_attrs.api_keys).append(tables.ApiKey(id=label, key=key_hash))
                    console.print(
                        f"  Datasource: [cyan]{datasource.name}[/cyan] "
                        f"(ID: {datasource.id}) "
                        f"[blue](API Key: {key})[/blue]"
                    )
                    for experiment in await datasource.awaitable_attrs.experiments:
                        console.print(f"    Experiment: [cyan]{experiment.name}[/cyan] (ID: {experiment.id})")
        except IntegrityError as err:
            await session.rollback()
            err_console.print(f"[bold red]Error:[/bold red] {err}")
            raise typer.Exit(1) from err


@app.command()
def create_nacl_keyset(
    output: Annotated[
        Base64OrJson,
        typer.Option(help="Output format. Use base64 when generating a key for use in an environment variable."),
    ] = Base64OrJson.base64,
):
    """Generate an encryption keyset for the "nacl" secret provider.

    The encoded encryption key will be written to stdout.

    When --output=base64 (default), the output can be used as the XNGIN_SECRETS_NACL_KEYSET environment variable.
    """
    from xngin.xsecrets.nacl_provider import NaclProviderKeyset  # noqa: PLC0415

    keyset = NaclProviderKeyset.create()
    if output == Base64OrJson.base64:
        print(keyset.serialize_base64())
    else:
        print(keyset.serialize_json())


@app.command()
def encrypt(
    aad: Annotated[
        str,
        typer.Option(help="Bind the ciphertext to this additionally authenticated data (AAD)."),
    ] = "cli",
):
    """Encrypts a string using the same encryption configuration that the API server does."""
    plaintext = sys.stdin.read()
    print(secretservice.get_symmetric().encrypt(plaintext, aad))


@app.command()
def decrypt(
    aad: Annotated[str, typer.Option(help="The AAD specified when the ciphertext was encrypted.")] = "cli",
):
    """Decrypts a string using the same encryption configuration that the API server does."""
    ciphertext = sys.stdin.read()
    print(secretservice.get_symmetric().decrypt(ciphertext, aad))


@app.command()
def generate_typed_clients():
    """Generates strongly typed API clients from the FastAPI definitions."""
    # dev-only dependency
    import fastapi_typed_client  # noqa: PLC0415

    root = Path("src/xngin/apiserver/testing")
    eapi_path = root / "experiments_api_client.py"
    aapi_path = root / "admin_api_client.py"
    iadminapi_path = root / "admin_integrations_api_client.py"
    iapi_path = root / "integrations_api_client.py"

    print(f"Generating ExperimentsAPIClient: {eapi_path}")
    fastapi_typed_client.generate_fastapi_typed_client(
        "xngin.apiserver.routers.experiments.experiments_api:router",
        include_security_params=True,
        output_path=eapi_path,
        raise_if_not_default_status=True,
        title="ExperimentsAPIClient",
    )
    print(f"Generating AdminAPIClient: {aapi_path}")
    fastapi_typed_client.generate_fastapi_typed_client(
        "xngin.apiserver.routers.admin.admin_api:router",
        output_path=aapi_path,
        raise_if_not_default_status=True,
        title="AdminAPIClient",
    )
    print(f"Generating AdminIntegrationsAPIClient: {iadminapi_path}")
    fastapi_typed_client.generate_fastapi_typed_client(
        "xngin.apiserver.routers.admin_integrations.admin_integrations_api:router",
        output_path=iadminapi_path,
        raise_if_not_default_status=True,
        title="AdminIntegrationsAPIClient",
    )

    print(f"Generating IntegrationsAPIClient: {iapi_path}")
    fastapi_typed_client.generate_fastapi_typed_client(
        "xngin.apiserver.routers.integrations.integrations_api:router",
        include_security_params=True,
        output_path=iapi_path,
        raise_if_not_default_status=True,
        title="IntegrationsAPIClient",
    )

    ruff_bin = shutil.which("ruff")
    if ruff_bin is None:
        return

    print("Formatting generated files...")
    try:
        subprocess.run(  # noqa: S603
            [
                ruff_bin,
                "format",
                eapi_path,
                aapi_path,
                iadminapi_path,
                iapi_path,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        err_console.print(f"[bold red]Error:[/bold red] ruff formatting failed: {exc}")
        raise typer.Exit(1) from exc


@snapshots_app.command("create-fake")
def snapshots_create_fake(
    dsn: Annotated[str, typer.Option("--dsn", "-d", help="Database connection string", envvar="DATABASE_URL")],
    exp_id: Annotated[str, typer.Option("--exp-id", help="Experiment ID")],
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", "-s", help="Start date in ISO format. Defaults to now."),
    ] = None,
    n: Annotated[int, typer.Option("--n", "-n", help="Number of daily snapshots to create.")] = 1,
    arm_id: Annotated[str | None, typer.Option("--arm-id", "-a", help="Arm ID to apply values to")] = None,
    metric: Annotated[str | None, typer.Option("--metric", "-m", help="Metric name to apply values to")] = None,
    field: Annotated[
        str | None,
        typer.Option("--field", "-f", help="Field name to override in generated analyses."),
    ] = None,
    values: Annotated[list[float] | None, typer.Argument(help="Optional values to cycle through")] = None,
    random_seed: Annotated[int | None, typer.Option("--random-seed", "-r", help="Random seed")] = None,
    echo: Annotated[bool, typer.Option("--echo", help="Echo SQL queries")] = False,
) -> None:
    """Create fake snapshots for a frequentist experiment."""
    from xngin.apiserver.snapshots.fake_data import (  # noqa: PLC0415
        VALID_SNAPSHOT_FIELDS,
        create_fake_snapshots,
        get_arm_ids,
        get_freq_experiment_for_cli,
        get_metric_names,
    )

    engine = create_engine(dsn, connect_args={"application_name": CLI_DB_APPLICATION_NAME}, echo=echo)

    with Session(engine) as session:
        try:
            experiment = get_freq_experiment_for_cli(session, exp_id)
        except ValueError as err:
            err_console.print(f"Error: {err}")
            raise typer.Exit(1) from err

        if metric and metric not in get_metric_names(experiment):
            err_console.print(
                f"Error: metric '{metric}' not found in experiment. Available: {get_metric_names(experiment)}"
            )
            raise typer.Exit(1)

        if arm_id and arm_id not in get_arm_ids(experiment):
            err_console.print(f"Error: arm_id '{arm_id}' not found in experiment. Available: {get_arm_ids(experiment)}")
            raise typer.Exit(1)

        if field and field not in VALID_SNAPSHOT_FIELDS:
            err_console.print(f"Error: field '{field}' not valid. Must be one of: {VALID_SNAPSHOT_FIELDS}")
            raise typer.Exit(1)

        snapshots = create_fake_snapshots(
            session,
            experiment,
            start_date=parse_iso_datetime(start_date),
            n=n,
            arm_id=arm_id,
            metric_name=metric,
            field=field,
            values=values,
            random_seed=random_seed,
        )
        session.commit()

    print(f"Successfully created {len(snapshots)} snapshots for experiment {exp_id}")


if __name__ == "__main__":
    app()
