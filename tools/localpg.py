#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer==0.15.1",
#     "rich==13.7.0",
#     "psycopg==3.1.18",
# ]
# ///
#
# Note: this script generated mostly by Anthropic Claude.
#
import subprocess
import time

import psycopg
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme

# Create a theme with colors that work well in both light and dark modes
theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "command": "blue",
    "url": "green",
})

console = Console(theme=theme)

app = typer.Typer()

# PostgreSQL configuration parameters
PG_CONFIG = {
    "checkpoint_completion_target": "0.9",
    "checkpoint_timeout": "30min",
    "effective_cache_size": "3GB",
    "effective_io_concurrency": "200",
    "fsync": "off",
    "full_page_writes": "off",
    "huge_pages": "off",
    "maintenance_work_mem": "256MB",
    "max_connections": "20",
    "max_parallel_maintenance_workers": "2",
    "max_parallel_workers": "2",
    "max_parallel_workers_per_gather": "2",
    "max_wal_size": "6GB",
    "max_worker_processes": "4",
    "min_wal_size": "2GB",
    "random_page_cost": "1",
    "shared_buffers": "2GB",
    "synchronous_commit": "off",
    "temp_buffers": "1GB",
    "wal_buffers": "16MB",
    "work_mem": "26214kB",
}


def build_docker_command(daemon: bool, tmpfs: bool, port: int, name: str) -> list[str]:
    """
    Build the Docker command with the specified options.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{port}:5432",
        "-e",
        "POSTGRES_PASSWORD=postgres",
        "-e",
        "POSTGRES_DB=postgres",
    ]

    # Add daemon flag if specified
    if daemon:
        cmd.append("-d")

    # Add tmpfs mount if specified
    if tmpfs:
        cmd.extend(["--tmpfs", "/var/lib/postgresql/data:rw,noexec,nosuid,size=4G"])

    # Add postgres image and version
    cmd.append("postgres:16")

    # Add all PostgreSQL configuration parameters
    for key, value in PG_CONFIG.items():
        cmd.extend(["-c", f"{key}={value}"])

    return cmd


@app.command()
def run(
    daemon: bool = typer.Option(
        False,
        "--daemon",
        "-d",
        help="Run Postgres in the background",
        envvar="LOCALPG_DAEMON",
    ),
    tmpfs: bool = typer.Option(
        False,
        "--tmpfs",
        "-t",
        help="Mount a tmpfs volume for data directory",
        envvar="LOCALPG_TMPFS",
    ),
    port: int = typer.Option(
        5499, "--port", "-p", help="Port to expose PostgreSQL on", envvar="LOCALPG_PORT"
    ),
    name: str = typer.Option(
        "localpg",
        "--name",
        "-n",
        help="Name of the Docker container",
        envvar="LOCALPG_NAME",
    ),
    allow_existing: bool = typer.Option(
        False,
        "--allow-existing",
        help="Exit successfully if container already exists",
        envvar="LOCALPG_ALLOW_EXISTING",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Wait for port to be available (up to 30 seconds)",
        envvar="LOCALPG_WAIT",
    ),
    create_db: str = typer.Option(
        None,
        "--create-db",
        help="Create a new database with this name (requires -d/--daemon)",
        envvar="LOCALPG_CREATE_DB",
    ),
    drop_db_first: bool = typer.Option(
        False,
        "--drop-db-first",
        help="Drop the database specified by --create-db if it already exists",
        envvar="LOCALPG_DROP_DB_FIRST",
    ),
    if_created: str = typer.Option(
        None,
        "--if-created",
        help="Shell command to run if the database specified by --create-db doesn't already exist",
        envvar="LOCALPG_IF_CREATED",
    ),
):
    """
    Start a local ephemeral PostgreSQL instance using Docker.
    """
    if create_db and (not daemon or not wait):
        console.print("[warning]--create-db requires --daemon and --wait[/]\n")
        raise typer.Exit(3)

    try:
        # Check if container already exists
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name=^{name}$",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            console.print(f"⚠️ [warning]Container named '{name}' already exists.[/]")
            if allow_existing:
                console.print("[info]Continuing due to --allow-existing flag[/]")
            else:
                console.print(
                    f"[command]Run 'docker kill {name}' to remove it first.[/]\n"
                )
                # ruff: noqa: TRY301
                raise typer.Exit(3)
        else:
            cmd = build_docker_command(daemon, tmpfs, port, name)
            console.print(f"\n🔄 [info]Running command: {' '.join(cmd)}[/]")

            if not daemon:
                console.print(
                    "\n💡 [info]Press Ctrl+C to stop the container when done[/]\n"
                )

            subprocess.run(cmd, check=True)

            if daemon:
                if wait:
                    wait_for_postgres(port)

                console.print("🚀 [info]Postgres started in daemon mode.[/]")
                console.print(
                    f"\n⚡ To stop the container, run:\n   [command]docker kill {name}[/]"
                )
                console.print(
                    f"\n🪓 To view the logs, run:\n   [command]docker logs --follow {name}[/]"
                    f"\n🔌 Default connection string:\n   [url]postgresql://postgres:postgres@localhost:{port}/postgres?sslmode=disable[/]"
                )

        # Create the database if requested even if the container was already running.
        if daemon and create_db:
            db_created = create_database(create_db, port, drop_db_first)
            if db_created and if_created:
                console.print(f"\n🔄 [info]Running command: {if_created}[/]")
                try:
                    subprocess.run(if_created, shell=True, check=True)
                    console.print("✅ [info]Command completed successfully[/]")
                except subprocess.CalledProcessError as e:
                    console.print(
                        f"❌ [warning]Command failed with exit code {e.returncode}[/]"
                    )
            console.print(
                f"🔌 Connection string:\n   [url]postgresql://postgres:postgres@localhost:{port}/{create_db}?sslmode=disable[/]"
            )
        console.print("")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running Docker command: {e}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1) from e


def create_database(dbname: str, port: int, drop_database: bool = False) -> bool:
    """Creates a new database using psycopg.

    Returns:
        bool: True if database was created, False if it already existed
    """
    # Connect to the default postgres database to create all our custom databases.
    conn_str = f"postgresql://postgres:postgres@localhost:{port}/postgres"
    try:
        with psycopg.connect(conn_str, autocommit=True) as conn:
            if drop_database:
                conn.execute(f"DROP DATABASE IF EXISTS {dbname}")
            conn.execute(f"CREATE DATABASE {dbname}")
            console.print(f"\n✨ [info]Created database '{dbname}'[/]")
            return True
    except psycopg.errors.DuplicateDatabase:
        console.print(f"⚠️ [warning]Database '{dbname}' already exists[/]")
        return False


def wait_for_postgres(port):
    """Wait for PostgreSQL to be ready to accept connections."""
    conn_str = f"postgresql://postgres:postgres@localhost:{port}/postgres"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting for PostgreSQL to be ready...", total=None)
        start_time = time.time()
        while time.time() - start_time < 30:  # 30 second timeout
            try:
                psycopg.connect(conn_str, connect_timeout=1).close()
                progress.update(task, completed=True)
                break
            except psycopg.OperationalError:
                time.sleep(0.25)  # 250ms between retries
        else:
            console.print(
                "\n❌ [warning]Timed out waiting for PostgreSQL to start[/]\n"
            )
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
