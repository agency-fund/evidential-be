"""Helpers shared by `cli/main.py` and the per-command modules under `cli/commands/`."""

import os
import stat
from pathlib import Path
from typing import NoReturn

import sqlalchemy
import typer
from rich.console import Console
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from xngin.apiserver.dwh import dwh_utils

SA_LOGGER_NAME_FOR_CLI = "cli_dwh"

CLI_DB_APPLICATION_NAME = f"cli-{os.getpid()}"

OWNER_ONLY_MODE = stat.S_IRUSR | stat.S_IWUSR

console = Console()
err_console = Console(stderr=True)


def write_file_atomically(path: Path, content: str, *, private: bool = False) -> None:
    """Writes content to path atomically.

    Pass private=True for files holding secrets: the file is restricted to the current user, as if it had been
    created and then `chmod og=`ed. This only has effect on Unix-like filesystems.
    """
    if path.exists() and not path.is_file():
        # Character devices like /dev/null can't be renamed over, and atomicity means nothing for them.
        path.write_text(content)
        return

    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if private:
            temp_path.touch(mode=OWNER_ONLY_MODE)
            temp_path.chmod(OWNER_ONLY_MODE)
        temp_path.write_text(content)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def fail(message: str, *, code: int = 1) -> NoReturn:
    """Prints an error to stderr and exits.

    Use code 2 to indicate user error.
    """
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code)


def cli_connect_args(url: sqlalchemy.URL) -> dict:
    """Returns the connect_args identifying this process to the server, for dialects that accept them.

    Only the Postgres drivers understand application_name.
    """
    if dwh_utils.is_postgres(url):
        return {"application_name": CLI_DB_APPLICATION_NAME}
    return {}


def cli_engine(url: sqlalchemy.URL | str, *, connect_args: dict | None = None, echo: bool = False) -> sqlalchemy.Engine:
    """Creates an Engine with the logging name, application name, and dialect workarounds the CLI expects."""
    url = sqlalchemy.make_url(url)
    engine = create_engine(
        url, connect_args=cli_connect_args(url) | (connect_args or {}), logging_name=SA_LOGGER_NAME_FOR_CLI, echo=echo
    )
    dwh_utils.extra_engine_setup(engine)
    return engine


def cli_async_engine(url: sqlalchemy.URL | str) -> AsyncEngine:
    """Creates an Engine comparable to what cli_engine would create, but async."""
    url = sqlalchemy.make_url(url)
    engine = create_async_engine(url, connect_args=cli_connect_args(url), logging_name=SA_LOGGER_NAME_FOR_CLI)
    dwh_utils.extra_engine_setup(engine.sync_engine)
    return engine
