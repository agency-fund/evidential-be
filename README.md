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
curl -LsSf https://astral.sh/uv/0.5.5/install.sh | sh
```

## Settings

The settings schema is defined in
[`xngin.apiserver.settings:XnginSettings`](src/xngin/apiserver/settings.py). Values are read at
startup from environment variables and from a JSON file specified by the `XNGIN_SETTINGS`
environment variable. This defaults to [xngin.settings.json](xngin.settings.json), but you will use
[xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) file during
most of your work.

The settings files can be committed to version control but secret values should not be. They should be referred to with
`${secret:NAME}` syntax. When the XNGIN_SECRETS_SOURCE environment variable is unset or set to "environ", those
references will be replaced with a corresponding environment variable value.

There are 3 levels of configuration behind Xngin:

* App-wide settings as noted above in `XnginSettings`. This includes per-client configuration allowing us to provide a
  multi-tenant service. It is retrieved in production via dependency injection (see
  [`settings.py:get_settings_for_server`](src/xngin/apiserver/settings.py)), which is overriden in tests (see
  [`conftest.py:get_settings_for_test`](src/xngin/apiserver/conftest.py)).
* Client-level configuration whose schema is [`ClientConfig`](src/xngin/apiserver/settings.py), which can be one of a
  fixed set of supported customer configurations (e.g. `RemoteDatabaseConfig`) that package up how to connect to the
  data warehouse (see `Dsn`) along with all the different `Participant` types (aka each type of unit of experimentation,
  e.g. a WhatsApp group, or individuals).
* Participant type-level configuration with schema [`config_sheet.py:ConfigWorksheet`](src/xngin/sheets/config_sheet.py), including column and type info derived from the warehouse via introspection (see
  [`config_sheet.py:create_configworksheet_from_table`](src/xngin/sheets/config_sheet.py),
  [`main.py:get_sqlalchemy_table_from_engine`](src/xngin/apiserver/main.py)), as well as extra metadata about columns
  (is_strata/is_filter/is_metric) in Google spreadsheets as specified by the client. Both sources (dwh introspection,
  gsheets) are represented by the `ConfigWorksheet` model, although not all information may be supplied by either.


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

   `pytest -rA` to print out _all_ stdout from your tests; `-rx` for just those failing. (See
   [docs](https://docs.pytest.org/en/latest/how-to/output.html#producing-a-detailed-summary-report) for more info.)

   Running the unit tests will create the testing_dwh.db database. This database will be used by your local development
   server.

3. Then start the dev server:

   ```shell
   uv run fastapi dev src/xngin/apiserver/main.py
   ```
   To change the port, add the flag `--port <myport>`.

4. Send some test requests:

   Each request should have an HTTP request header `Config-ID` set to the `id` value of a configuration entry in the
   [xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) settings file. For testing
   and local dev, use `testing`.

   ```shell
   curl -H "Config-ID: testing" 'http://localhost:8000/filters?participant_type=test_participant_type'
   curl -H "Config-ID: testing" 'http://localhost:8000/strata?participant_type=test_participant_type'
   ```

   Also see [apitest.strata.xurl](src/xngin/apiserver/testdata/apitest.strata.xurl) for a complete example of how to
   write an API test script.

5. Visit the local interactive docs page: http://localhost:8000/docs

6. `uv` sets up a virtual environment by default. To avoid needing to use `uv run` before commands installed by our
   project, activate the environment with:

   ```shell
   source .venv/bin/activate
   ```

   The environment var `VIRTUAL_ENV` should now be set, pointing to the project's environment.

   You can exit the virtual environment with `deactivate`.

7. Now set up the pre-commit hooks in your local git with:

   ```shell
   pre-commit install
   ```

   which will run the checks whenever you create or modify a commit. It only does checks against files that changed; to
   force it to check all files, you can run:

   ```
   pre-commit run -a
   ```
8. To parse a proper Google Sheets config, you'll need a service worker token, whose json info should be placed in `~/.config/gspread/service_account.json` by default.
   - [Setup a service account in your GCP console](console.cloud.google.com) > select your project via dropdown at the top > IAM & Admin > Service Accounts > + Create Service Account > give it a name, desc and create; *note the email addr created* > After creation, click the email for that account > Keys tab > Create the json key file and put it in the above location. Lastly, share the spreadsheet as Viewer-only with this special service account email address.
   - Ensure that the [Google Sheets API](  https://console.developers.google.com/apis/api/sheets.googleapis.com/overview) is enabled for your google cloud project.


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

## FAQ

### How do I build the Docker container locally?

```shell
docker build -t xngin .
docker run xngin:latest
```

See the next question for an example of how to pass settings to the container.

### How do I run the tests in the Docker container?

See [.github/workflows/test.yaml](.github/workflows/test.yaml).

### How do I run the Docker images built by the CI?

```shell
# Authenticate your local Docker to the ghcr using a GitHub Personal Access Token that has at least the `read:packages`
# scope. Create one here: https://github.com/settings/tokens
docker login ghcr.io -u YOUR_GITHUB_USERNAME
```

Then you can start the container locally with something like this:

```shell
docker run \
  -it \
  --env-file ./.env \
  -v path/to/settings/directory:/settings \
  -e XNGIN_SETTINGS=/settings/xngin.settings.json \
  ghcr.io/agency-fund/xngin:main
```

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

### How do I force-build the test sqlite database?

Recommend deleting the `src/xngin/apiserver/testdata/testing_dwh.db` and let the unit tests rebuild it
for you.

You could use the [CLI](src/xngin/cli/main.py) `create-testing-dwh` command (see --help for more):
```shell
uv run xngin-cli create-testing-dwh \
   --dsn sqlite:///src/xngin/apiserver/testdata/testing_dwh.db \
   --table-name=test_participant_type
```
BUT the data types used in the create ddl will differ right now as the former relies on pandas to infer types while the latter uses our own mapping based on dataframe types.

### How do I run xngin against a local Postgres?

Here's an example of how to run a local Postgres in Docker and have
xngin use it as the system database:

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

#### How can I load test data into this pg instance with a different schema?

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

You could run the individual component tests as defined in the various jobs under [test.yaml](.github/workflows/test.yaml), but to best replicate the environment as used by our GHA, we recommend you install [`act`](https://github.com/nektos/act). Then you could execute whole workflows or individual jobs within, e.g.:
```shell
# list all jobs across our different workflows
act -l
# run an particular job in a workflow (for the default "on: push" event)
act -W .github/workflows/test.yaml -j smoke-server
# equivalently:
act -j smoke-server
```

#### On Macs

* You might see this error:
```
Error: failed to start container: Error response from daemon: error while creating mount source path '/host_mnt/Users/me/.docker/run/docker.sock': mkdir /host_mnt/Users/me/.docker/run/docker.sock: operation not supported
```
If so, you can resolve it by adding this line to a `~/.actrc` file as noted in [this issue](https://github.com/nektos/act/issues/2239#issuecomment-2189419148):
`--container-daemon-socket=unix:///var/run/docker.sock`

* When running `-j unittests`, you can ignore this error line, as `act` doesn't support a macos runner:
```
[tests/Python on macOS-2] 🚧  Skipping unsupported platform -- Try running with `-P macos-14=...`
```
You can force `act` to [use your localhost as the runner](https://github.com/nektos/act/issues/97#issuecomment-1868974264) with the experimental platform `-P macos-14=-self-hosted` flag, but that's not advisable as you may overwrite local files such as service_account.json.

* [Run a particular matrix](https://github.com/nektos/act/pull/1675) configuration with e.g. `--matrix os:ubuntu-22.04`
So a more complex command that also injects a secret from a file (which could also be [placed in your .actrc](https://nektosact.com/usage/index.html?highlight=secret#envsecrets-files-structure)) might look like:
```act --matrix os:ubuntu-22.04 -j unittests -s GCLOUD_SERVICE_ACCOUNT_CREDENTIALS="$(< ~/.config/gspread/service_account.json)"
```

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
