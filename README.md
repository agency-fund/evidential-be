[![lint](https://github.com/agency-fund/xngin/actions/workflows/lint.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/lint.yaml)
[![precommit](https://github.com/agency-fund/xngin/actions/workflows/precommit.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/precommit.yaml)
[![test](https://github.com/agency-fund/xngin/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/test.yaml)

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

## Prerequisite

The commands below require you to have [uv](https://docs.astral.sh/uv/) installed:

```shell
curl -LsSf https://astral.sh/uv/0.5.1/install.sh | sh
```

## Settings

The settings schema is defined in [xngin.apiserver.settings](src/xngin/apiserver/settings.py). Values are read at
startup from environment variables and from a JSON file specified by the XNGIN_SETTINGS environment variable. This
defaults
to [xngin.settings.json](xngin.settings.json), but you will
use [xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) file during most of your
work.

The settings files can be committed to version control but secret values should not be. They should be referred to with
`=secret:NAME` syntax. Resolving these values is not yet implemented.

## Getting Started

Follow the steps below to get a local development environment running.

1. For local development, use the testing settings file:

   ```shell
   export XNGIN_SETTINGS=src/xngin/apiserver/testdata/xngin.testing.settings.json
   ```

2. Then run the unit tests:

   ```shell
   uv run pytest
   ```

   Running the unit tests will create the testing_dwh.db database. This database will be used by your local development
   server.

3. Then start the dev server:

   ```shell
   uv run fastapi dev src/xngin/apiserver/main.py
   ```

4. Send some test requests:

   Each request should have an HTTP request header `Config-ID` set to the `id` value of a configuration entry in the
   [xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) settings file. For testing
   and local dev, use `testing`.

   ```shell
   curl -H "Config-ID: testing" 'http://localhost:8000/filters?unit_type=test_unit_type'
   curl -H "Config-ID: testing" 'http://localhost:8000/strata?unit_type=test_unit_type'
   ```

   Also see [apitest.strata.hurl](src/xngin/apiserver/testdata/apitest.strata.hurl) for a complete example of how to
   write an API test script.

6. Visit the local interactive docs page: http://localhost:8000/docs

## FAQ

### How do I build and run via Docker?

> The Docker image is not useful for anything (yet).

```shell
docker build -t xngin .
docker run xngin:latest
```

### How do I add a Python dependency?

1. Add the dependency to [pyproject.toml](pyproject.toml) (replace httpx with whatever dependency you are adding). Try
   to pin it to a narrow version range, if possible.
   ```shell
   uv add httpx
   ```
2. Install the new dependencies into your environment:
    ```shell
    uv lock
    uv sync
    ```
3. Run the unit tests to ensure everything still works.
   ```shell
   uv run pytest
   ```
4. Commit the changed uv.lock and pyproject.toml files.
5.

### psycopg2 module does not install correctly.

You might see this error:

> Error: pg_config executable not found.
>
> pg_config is required to build psycopg2 from source.

The fix will depend on your specific environment.

#### Linux

1. If on Linux, try: `sudo apt install -y libpq-dev` and then re-install dependencies.
2. See https://www.psycopg.org/docs/install.html.
3. See https://www.psycopg.org/docs/faq.html.

#### OSX

Run `brew install postgresql@14`.
