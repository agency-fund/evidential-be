# xngin

Python version of [RL Experiments Engine](https://github.com/agency-fund/rl-experiments-engine).

The following is a proposal of the main components of this service:

1. A ODBC/DBI-based interfae module to connect to underlying data sources (one table per unit of analysis)
2. A configuration module that draws from the table(s) specified in (1) into a Google Sheet that can be annotated with
   filters, metrics and strata
3. API endpoints that provide a list of fields and their values (/filters, /metrics, /strata)
4. API endpoints that provide a power analysis check and stratified random assignment of treatment
5. A treatment assignment service that stores treatment assignments and provides an API endpoint to provide treatment
   status by ID
6. Save experiment (inclusive of Audience) specifications

## Settings

The settings schema is defined in app/settings.py. Values are read at startup from environment variables and the
xngin.settings.json file. Unit tests use the xngin.testing.settings.json file.

xngin.settings.json can be committed to version control but secret values should not be. They should be referred to with
`=secret:NAME` syntax. Resolving these values is not yet implemented.

## Helpful Commands

## Prerequisite

The commands below require you to have [uv](https://docs.astral.sh/uv/) installed:

```shell
curl -LsSf https://astral.sh/uv/0.4.5/install.sh | sh
```

### Run Locally

This will expose the service at http://127.0.0.1:8000/docs with auto-reload features:

```shell
uv run fastapi dev src/xngin/apiserver/main.py
```

### Running Tests

```shell
uv run pytest
```

### Run via Docker

```shell
docker build -t xngin .
docker run xngin:latest
```

### I want to add a Python dependency.

1. Add the dependency to [pyproject.toml](pyproject.toml).
2. Update the lockfile:
    ```shell
    uv lock
    ```
3. Rebuild your local docker container (see above).
4. Install it into your environment:
    ```shell
    uv sync
    ```

### psycopg2 module does not install correctly.

You might see this error:

> Error: pg_config executable not found.
>
> pg_config is required to build psycopg2 from source.

The fix will depend on your specific environment, but these will be helpful:

1. If on Linux, try: `sudo apt install -y libpq-dev` and then re-install dependencies.
2. See https://www.psycopg.org/docs/install.html.
3. See https://www.psycopg.org/docs/faq.html.

## Sample configuration spreadsheets

Simplified RL config:
https://docs.google.com/spreadsheets/d/redacted/edit?gid=1616931447#gid=1616931447

Config sheet corresponding to dwh.csv:
https://docs.google.com/spreadsheets/d/redacted/edit?gid=0#gid=0
