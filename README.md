[![lint](https://github.com/agency-fund/xngin/actions/workflows/lint.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/lint.yaml)
[![precommit](https://github.com/agency-fund/xngin/actions/workflows/precommit.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/precommit.yaml)
[![test](https://github.com/agency-fund/xngin/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/test.yaml)

# xngin

- [xngin](#xngin)
  - [Prerequisites](#prerequisites)
  - [Getting Started](#getting-started)
  - [Settings](#settings)
  - [Docker](#docker)
  - [Testing](#testing)
  - [The CLI](#the-cli)
  - [Onboarding new Clients](#onboarding-new-clients)
  - [FAQ](#faq)
  - [Deployment on Railway](#deployment-on-railway)
  - [Admin API](#admin-api)
  - [OIDC](#oidc)
  - [API Keys](#api-keys)
  - [Schema Migration](#schema-migration)

Python version of [RL Experiments Engine](https://github.com/agency-fund/rl-experiments-engine).

The following is a proposal of the main components of this service:

1. A ODBC/DBI-based interface module to connect to underlying data sources (one table per unit of analysis)
2. A configuration module that draws from the table(s) specified in (1) into a Google Sheet that can be annotated with
   filters, metrics and strata
3. API endpoints that provide a list of fields and their values (/filters, /metrics, /strata)
4. API endpoints that provide a power analysis check and stratified random assignment of treatment
5. A treatment assignment service that stores treatment assignments and provides an API endpoint to provide treatment
   status by ID
6. Save experiment (inclusive of Audience) specifications

## Prerequisites

The commands below require you to have [uv](https://docs.astral.sh/uv/) installed:

```shell
curl -LsSf https://astral.sh/uv/0.5.14/install.sh | sh
```

## Getting Started

Follow the steps below to get a local development environment running.

1. Update the project's environment, install dependencies, and create a virtual environment (.venv).
  ```shell
  uv sync
  ```

2. For local development, use the testing settings file:

   ```shell
   export XNGIN_SETTINGS=src/xngin/apiserver/testdata/xngin.testing.settings.json
   ```

3. Then run the unit tests:

   ```shell
   uv run pytest
   ```

   `pytest -rA` to print out _all_ stdout from your tests; `-rx` for just those failing. (See
   [docs](https://docs.pytest.org/en/latest/how-to/output.html#producing-a-detailed-summary-report) for more info.)

   Running the unit tests will create the `testing_dwh.db` database. This database will be used by your local
   development
   server.

4. Then start the dev server:

   ```shell
   uv run fastapi dev src/xngin/apiserver/main.py
   ```
   To change the port, add the flag `--port <myport>`.

5. Send some test requests:

   Each request should have an HTTP request header `Datasource-ID` set to the `id` value of a configuration entry in the
   [xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) settings file. For testing
   and local dev, use `testing`.

   ```shell
   curl -H "Datasource-ID: testing" 'http://localhost:8000/filters?participant_type=test_participant_type'
   curl -H "Datasource-ID: testing" 'http://localhost:8000/strata?participant_type=test_participant_type'
   ```

   Also see [apitest.strata.xurl](src/xngin/apiserver/testdata/apitest.strata.xurl) for a complete example of how to
   write an API test script.

6. Visit the local interactive docs page: http://localhost:8000/docs

7. `uv` sets up a virtual environment by default. To avoid needing to use `uv run` before commands installed by our
   project, activate the environment with:

   ```shell
   source .venv/bin/activate
   ```

   The environment var `VIRTUAL_ENV` should now be set, pointing to the project's environment.

   You can exit the virtual environment with `deactivate`.

8. Now set up the pre-commit hooks in your local git with:

   ```shell
   pre-commit install
   ```

   which will run the checks whenever you create or modify a commit. It only does checks against files that changed; to
   force it to check all files, you can run:

   ```
   pre-commit run -a
   ```

9. To parse a proper Google Sheets config, you'll need a service worker token, whose json info should be placed in
   `~/.config/gspread/service_account.json` by default.
    - [Setup a service account in your GCP console](console.cloud.google.com) > select your project via dropdown at the
      top > IAM & Admin > Service Accounts > + Create Service Account > give it a name, desc and create; *note the email
      addr created* > After creation, click the email for that account > Keys tab > Create the json key file and put it
      in the above location. Lastly, share the spreadsheet as Viewer-only with this special service account email
      address.
    - Ensure that
      the [Google Sheets API](  https://console.developers.google.com/apis/api/sheets.googleapis.com/overview) is
      enabled for your google cloud project.

### Learn more

Regarding some of the python libraries and features we use, see:

* [Pydantic concepts](https://docs.pydantic.dev/2.8/concepts/models/) for defining model schemas with input parsing and
  coercion, [custom validation](https://docs.pydantic.dev/2.8/concepts/validators/) and
  custom [serialization](https://docs.pydantic.dev/2.8/concepts/serialization/) support as needed. Also be aware of its
  use of `Annotated` to add metadata that modify how types are validated, serialized,
  etc. [[1](https://docs.pydantic.dev/2.8/concepts/fields/#using-annotated), [2](https://docs.pydantic.dev/2.8/concepts/types/#composing-types-via-annotated)].
* [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/first-steps/) - skim this for key concepts around defining "
  path operations" and [adding metadata using
  `Annotated`](https://fastapi.tiangolo.com/python-types/#type-hints-with-metadata-annotations) for key components (e.g.
  Query, Path, ...) to add extra documentation or do additional validation (internally using Pydantic). Also read up on
  how it does [dependency injection](https://fastapi.tiangolo.com/tutorial/dependencies/) with its `Depends` metadata,
  which we use (see [main.py](src/xngin/apiserver/main.py) and [dependencies.py](src/xngin/apiserver/dependencies.py)).
  FastAPI also generates OpenAPI documentation for us under the server's `/docs` endpoint, leveraging Pydantic data
  models to generate the schemas.
* [SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/tutorial/index.html) - is used to allow more uniform
  interaction with a variety of database dialects via its DBAPI interface and ORM features. These include client data
  warehouses as well as our own application store (e.g. for caching client configuration of their tables).
* Other dev tooling: [mypy](https://mypy-lang.org/) for static type checking (configured in
  `pyproject.toml`), [ruff](https://github.com/astral-sh/ruff) for fast python linting and formatting (also see
  `pyproject.toml`), [pre-commit](https://pre-commit.com/) to automatically run a number of checks including ruff before
  your commit (see `.pre-commit-config.yaml`).

## Settings

Values are read at startup from environment variables and from a JSON file specified by the `XNGIN_SETTINGS` environment
variable. This defaults to [xngin.settings.json](xngin.settings.json), but you will use
[xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) file during most of your work.

The settings files can be committed to version control but secret values should not be. They should be referred to with
`${secret:NAME}` syntax. When the `XNGIN_SECRETS_SOURCE` environment variable is unset or set to "environ", those
references will be replaced with a corresponding environment variable value.

There are 3 levels of configuration behind Xngin:

* **App-wide** settings as noted above use a schema defined in
  [`xngin.apiserver.settings:XnginSettings`](src/xngin/apiserver/settings.py). This includes per-client configuration
  allowing us to provide a multi-tenant service. It is retrieved in production via dependency injection (see
  [`settings.py:get_settings_for_server`](src/xngin/apiserver/settings.py)), which is overriden in tests (see
  [`conftest.py:get_settings_for_test`](src/xngin/apiserver/conftest.py)).
* The above wraps **client-level** configuration whose schema is [`ClientConfig`](src/xngin/apiserver/settings.py),
  which
  can be one of a fixed set of supported customer configurations (e.g. `RemoteDatabaseConfig`) that package up how to
  connect to the data warehouse (see `BaseDsn` and descendants) along with all the different `Participant` types (aka
  each type of unit of experimentation, e.g. a WhatsApp group, or individual phone numbers, hospitals, schools, ...).
* **Participant type-level** configuration with schema
  [`config_sheet.py:ConfigWorksheet`](src/xngin/sheets/config_sheet.py), including column and type info derived from the
  warehouse via introspection (see
  [`config_sheet.py:create_configworksheet_from_table`](src/xngin/sheets/config_sheet.py),
  [`main.py:get_sqlalchemy_table_from_engine`](src/xngin/apiserver/main.py)), as well as extra metadata about columns
  (is_strata/is_filter/is_metric). The extra metadata may come from CSV or in Google spreadsheets as filled out by the
  client. Both sources (dwh introspection, gsheets) are represented by the `ConfigWorksheet` model, although not all
  information may be supplied by either.
    * This information is also cached in our app (system) db as specified in
    * [`database.py`](src/xngin/apiserver/database.py). The db DSN can be overriden via the `XNGIN_DB` environment
      variable to point to something other than the default sqlite database `xngin.db`, which otherwise is created at
      this
      root level.

## Docker

### How do I build and run the Docker container locally?

```shell
docker build -t xngin .
docker run \
  -it \
  --env-file secrets/.env \
  -v `pwd`/settings/:/settings \
  -e XNGIN_SETTINGS=/settings/xngin.settings.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/settings/gs_service_account.json \
  -p 8000:8000  \
   xngin:latest
```

### How do I run the Docker images [built by the CI](https://github.com/agency-fund/xngin/pkgs/container/xngin)?

```shell
# Authenticate your local Docker to the ghcr using a GitHub Personal Access Token (classic) that has at least the
# `read:packages` scope. Create one here: https://github.com/settings/tokens
docker login ghcr.io -u YOUR_GITHUB_USERNAME
```

Then you can start the container locally with something like this:

```shell
docker run \
  -it \
  --env-file ./.env \
  -v path/to/settings/directory:/settings \
  -e XNGIN_SETTINGS=/settings/xngin.settings.json \
  -p 127.0.0.1:8000:8000 \
  ghcr.io/agency-fund/xngin:main
```

This also means we can fetch prod settings from elsewhere on our host and mount the dir with it in the container.
Don't forget to add other environment variables not set on the command line to a `.env` file such as
GOOGLE_APPLICATION_CREDENTIALS; these are not part of the image. Also beware that Docker
[includes double quotes in the value of your env variables](https://github.com/docker/compose/issues/3702)
read in via --env-file.

### How do I run the tests in the Docker container?

See [.github/workflows/test.yaml](.github/workflows/test.yaml).

### How do I run xngin against a local Postgres running in Docker?

Here's an example of how to run a local Postgres and have xngin use it as the system database:

> Note: This example creates a Postgres database for the /system/ database; this is used only for caching
> configuration spreadsheets. If you want to test a customer database with postgres, you must edit the settings JSON.

```shell
PASSWORD="secret$(cat /dev/urandom | head -c128 | sha256sum | cut -b1-16)"
docker run --rm -d --name xngin-postgres \
  -e POSTGRES_USER=xnginwebserver \
  -e POSTGRES_PASSWORD=${PASSWORD} \
  -e POSTGRES_DB=xngin \
  -p 5432:5432 \
  -d postgres:17
export XNGIN_SETTINGS=src/xngin/apiserver/testdata/xngin.testing.settings.json
export XNGIN_DB=postgresql://xnginwebserver:${PASSWORD}@localhost:5432/xngin
uv run fastapi dev src/xngin/apiserver/main.py
```

## Testing

Run unittests with [pytest](https://docs.pytest.org/en/stable/).  `test_api.py` tests that use `testdata/*.xurl` data
can be updated more easily as things change by prefixing your pytest run with the environment variable:
`UPDATE_API_TESTS=1`.

[Smoke tests](.github/workflows/test.yaml) are also run as part of our github action test workflow.

* Some of our tests that rely on `conftest.py` will create a local sqlite db for testing in
  `src/xngin/apiserver/testdata/testing_dwh.db` if it doesn't exist already using the zipped data
  dump in `testing_dwh.csv.zst`.
* `testing_sheet.csv` is the corresponding spreadsheet that simulates a typical table configuration
  for the participant type data above.
* Our pytests have a test marked as 'integration' which is also only run as part of that workflow.
  To run, ensure you have the test credentials to access the gsheet (setting env var
  `GOOGLE_APPLICATION_CREDENTIALS` as necessary) then do:
   ```shell
   pytest -m integration
   ```

### How do I force-build the test sqlite database?

Recommend deleting the `src/xngin/apiserver/testdata/testing_dwh.db` and let the unit tests rebuild it
for you.

You could use the [CLI](src/xngin/cli/main.py) `create-testing-dwh` command (see --help for more):

```shell
uv run xngin-cli create-testing-dwh \
   --dsn sqlite:///src/xngin/apiserver/testdata/testing_dwh.db \
   --table-name=test_participant_type
```

BUT the data types used in the create ddl will differ right now as the former relies on pandas to infer types while the
latter uses our own mapping based on dataframe types.

### How can I load test data into my pg instance with a different schema?

As with a sqlite db above, use the CLI command:

```shell
uv run xngin-cli create-testing-dwh \
   --dsn=$XNGIN_DB \
   --password=$PASSWORD \
   --table-name=test_participant_type \
   --schema-name=alt
```

Now you can edit your `XNGIN_SETTINGS` json to add a ClientConfig that points to your local pg and
new table.

One way to manually query pg is using the `psql` terminal included with Postgres, e.g.:

```shell
psql -h localhost -p 5432 -d xngin -U xnginwebserver -c "select count(*) from alt.test_participant_type"
```

### How do I run our Github Action smoke tests?

You could run the individual component tests as defined in the various jobs
under [test.yaml](.github/workflows/test.yaml), but to best replicate the environment as used by our GHA, we recommend
you install [`act`](https://github.com/nektos/act). Then you could execute whole workflows or individual jobs within,
e.g.:

```shell
# list all jobs across our different workflows
act -l
# run an particular job in a workflow (for the default "on: push" event)
act -W .github/workflows/test.yaml -j smoke-server
# equivalently:
act -j smoke-server
```

To inject secrets, (e.g. `${{ secrets.GOOGLE_APPLICATION_CREDENTIALS_CONTENT }}`), supply it with the `-s` flag. This is
necessary to provide credentials for Google services, or else the tests that need them will fail. So to run the
unittests job successfully, do something like:

```shell
act -j unittests -s GOOGLE_APPLICATION_CREDENTIALS_CONTENT="$(< settings/service_account.json)"
```

#### On Macs

* You might see this error:

```
Error: failed to start container: Error response from daemon: error while creating mount source path '/host_mnt/Users/me/.docker/run/docker.sock': mkdir /host_mnt/Users/me/.docker/run/docker.sock: operation not supported
```

If so, you can resolve it by adding this line to a `~/.actrc` file as noted
in [this issue](https://github.com/nektos/act/issues/2239#issuecomment-2189419148):
`--container-daemon-socket=unix:///var/run/docker.sock`

* When running `-j unittests`, you can ignore this error line, as `act` doesn't support a macos runner:

```
[tests/Python on macOS-2] 🚧  Skipping unsupported platform -- Try running with `-P macos-14=...`
```

You can force `act`
to [use your localhost as the runner](https://github.com/nektos/act/issues/97#issuecomment-1868974264) with the
experimental platform `-P macos-14=-self-hosted` flag, but that's not advisable as you may overwrite local files such as
service_account.json.

* [Run a particular matrix](https://github.com/nektos/act/pull/1675) configuration with e.g. `--matrix os:ubuntu-22.04`
  So a more complex command that also injects a secret from a file (which could also
  be [placed in your .actrc](https://nektosact.com/usage/index.html?highlight=secret#envsecrets-files-structure)) might
  look like:

```shell
act --matrix os:ubuntu-22.04 -j unittests -s GOOGLE_APPLICATION_CREDENTIALS_CONTENT="$(< ~/.config/gspread/service_account.json)"
```

## The CLI

Helper tool to bootstap a new user and other operations such creating test data and validating configs. See the source
in `src/xngin/cli/main.py` and run:

```shell
uv run xngin-cli --help
```

## Onboarding new Clients

1. Get credentials to the client's data warehouse that has at least read-only access to the schemas/datasets
   containing the table(s) of interest. Each table will be a different "participant type" the user wishes to experiment
   over, and should contain a) a unique id column, b) features to filter the partipcants with (i.e. target for
   experiment
   eligibility), and c) metrics to use as possible outcomes to track, and optionally d) features to stratify on.

1. Generate the participant-level column metadata. This will ultimately be a google sheet the user can configure.
    1. First bootstrap column names and types from the dwh schema. There will be one row output per column in the target
       dwh table. See the command `uv run xngin-cli bootstrap-spreadsheet --help`
        1. If output as csv, import it to a new google spreadsheet that we the service provider will own.
        1. Share it with our gsheet service account.
    1. Share it with the client to mark which columns are filters/metrics/strata and which to use as the unique_id.
    1. Additional table columns can be added (or removed) in the future by the user.

1. Generate the client's config block in `xngin.settings.json`. Give them a unique string `"id:"` that they will pass
   back to us with every API request, specify `"type: "remote"` as the general type of dwh (see `settings.py`), provide
   dwh
   connectivity info in `"dwh":`, and lastly create the `"participants:"` list. Each item is a Participant object with a
   `"participant_type":` identifier for use in API requests, a `"table_name"` to look up in their dwh, and the GSheets
   URL
   and worksheet tab name from above to find its associated column metadata.

1. Bootstrap a set of data in the warehouse to use as a new Participant type.

For more examples, see the `xngin.gha.settings.json` settings used for testing.

### Supported DWHs and DSN url format

* Redshift - `postgresql+psycopg2://username@host:port/databasename`
* Postgres - `postgresql+psycopg://username@host:port/databasename`
* BigQuery (experimental) - `bigquery://some-project/some-dataset`
* SQLite3 (for tests) - `sqlite:///file_path`

## FAQ

> Note: These flags change depending on runtime environment. If running under an orchestrator, omit the -it. To run in
> the background, omit `-it` and add `-d`.

### How do I see the sql commands being executed in my logs?

Set `ECHO_SQL=1` in your environment, e.g.:

```shell
ECHO_SQL=1 XNGIN_SETTINGS=xngin.settings.json \
   uv run fastapi dev src/xngin/apiserver/main.py --port 8144
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

### psycopg2 module does not install correctly.

You might see this error:

> Error: pg_config executable not found.
>
> pg_config is required to build psycopg2 from source.

The fix will depend on your specific environment.

### BigQuery Support

BigQuery support is implemented but has not yet been fully tested.
See [.github/workflows/test.yaml](.github/workflows/test.yaml) for lifecycle tests.

Authentication:

* Only service account authentication is supported.
* The xngin-cli tools will authenticate via the GOOGLE_APPLICATION_CREDENTIALS environment variable. Example:
    ```shell
    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gspread/service_account.json
    xngin-cli create-testing-dwh --dsn 'bigquery://xngin-development-dc/ds'
    ```
* The xngin server authenticates using credentials specified in the settings file.
  See [xngin.gha.settings.json](xngin.gha.settings.json) for an example.

> Note: The GHA service account has permissions to access the xngin-development-dc.ds dataset.

> Note: You do not need the gcloud CLI or related tooling installed.

#### Linux

1. If on Linux, try: `sudo apt install -y libpq-dev` and then re-install dependencies.
2. See https://www.psycopg.org/docs/install.html.
3. See https://www.psycopg.org/docs/faq.html.

#### OSX

Run `brew install postgresql@14`.

## Deployment on Railway

The Railway deployment relies on [Dockerfile.railway](Dockerfile.railway), [railway.json](railway.json), and some
environment variables:

| Environment Variable         | Purpose                                                                                                                                                                                                    | Example                                                                                               |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| XNGIN_SETTINGS               | URL of the settings to fetch. This may be a private repository.                                                                                                                                            | https://api.github.com/repos/agency-fund/xngin-settings/contents/xngin.railway.settings.json?ref=main |
| XNGIN_SETTINGS_AUTHORIZATION | The value of the "Authorization:" header sent on the request to fetch XNGIN_SETTINGS. For GitHub URLs, this token requires read access to the content of the repository and must be prefixed with `token`. | `token ghp_....`                                                                                      |                                                                             |
| SENTRY_DSN                   | The Sentry ingestion endpoint. If unset, Sentry will not be configured for this instance. For TAF instances, see https://agency-fund.sentry.io/settings/projects/xngin/keys/.                              | https://...@...ingest.us.sentry.io/...                                                                |
| ENVIRONMENT                  | Declares an "environment" label for the runtime environment. Used by Sentry.                                                                                                                               | xngin-main.railway.app                                                                                |

In addition, there are variables set in the Railway console corresponding to configuration values referenced by
the [xngin.railway.settings.json](https://github.com/agency-fund/xngin-settings/xngin.railway.settings.json) file in the
limited-access https://github.com/agency-fund/xngin-settings repository.

## Admin API

The Admin API allows API keys to be managed by users from a trusted domain. The API is protected by OIDC and allows
logins from Google Workspace accounts in the @agency.fund domain.

The API is configured with environment variables:

| Environment Variable | Purpose                                                                   | Example |
|----------------------|---------------------------------------------------------------------------|---------|
| ENABLE_OIDC          | Enables the OIDC endpoints. Must be `true` for the Admin API to function. | `true`  |
| ENABLE_ADMIN         | Enables the Admin API.                                                    | `true`  |

If the Admin API is enabled, OIDC must be configured with additional environment variables (see below).

## OIDC

Our OIDC implementation supports the popup-style OIDC flow (response_type=`id_token`) and PKCE exchanges
(response_type=`code`).

| Environment Variable      | Purpose                                                                                                                                                                                                                                                           | Example                              |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| GOOGLE_OIDC_CLIENT_ID     | The Google-issued client ID.                                                                                                                                                                                                                                      | `2222-...apps.googleusercontent.com` |
| GOOGLE_OIDC_CLIENT_SECRET | The Google-generated client secret. Only required for PKCE.                                                                                                                                                                                                       | `G....`                              |
| GOOGLE_OIDC_REDIRECT_URI  | The URI that Google will redirect the user to after successfully authorizing. This should match the value configured in the Google Cloud console credential settings and the value embedded in the SPA. This is generally not used by the popup-style auth flows. | `http://localhost:8000/a/oidc`       |

## API Keys

The datasources defined in settings can be protected with API keys. To require API keys for a specific datasource, set
the `require_api_key` flag to true on the settings. Example:

```json5
{
  "id": "my-secure-config",
  "require_api_key": true,
  "config": {
    "type": "remote",
    // ...
  }
}
```

Use the Admin API to create API keys.

Clients must send the API keys as the `x-api-key` header. Example:

```shell
curl --header "x-api-key: xat_..." \
  --header "Datasource-ID: my-secure-config" \
  'http://localhost:8000/filters?participant_type=test_participant_type'
```

## Schema Migration

We are using [Atlas](https://atlasgo.io/) to manage database schema migrations.

To generate migrations:

```shell
uv run atlas migrate diff --env sa_postgres
```

To apply migrations to a local Postgres instance:

```shell
uv run atlas migrate apply --env sa_postgres --url 'postgresql://postgres:postgres@localhost:5499/postgres?sslmode=disable'
```

To apply migrations to the main.dev.agencyfund.org instance:

```shell
uv run atlas migrate apply --env sa_postgres --url 'postgresql://junction.proxy.rlwy.net:21126/railway?sslmode=require'
```
