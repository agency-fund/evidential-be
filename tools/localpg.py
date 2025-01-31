#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer==0.15.1",
#     "rich==13.7.0",
# ]
# ///
#
# Note: this script generated mostly by Anthropic Claude.
#
import typer
import subprocess
from rich.console import Console
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
        f"{port}:5432",
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
):
    """
    Start a local ephemeral PostgreSQL instance using Docker.
    """
    try:
        cmd = build_docker_command(daemon, tmpfs, port, name)

        if not daemon:
            console.print(
                "\n💡 [info]Press Ctrl+C to stop the container when done[/]\n"
            )

        subprocess.run(cmd, check=True)

        if daemon:
            console.print("\n🚀 [info]Postgres started in daemon mode.[/]")
            console.print(
                f"\n⚡ To stop the container, run:\n   [command]docker kill {name}[/]"
            )
            console.print(
                f"\n🪓 To view the logs, run:\n   [command]docker logs --follow {name}[/]"
            )
            console.print(
                f"\n🔌 Connection string:\n   [url]postgresql://postgres:postgres@localhost:{port}/postgres?sslmode=disable[/]"
            )
            console.print("")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running Docker command: {e}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
