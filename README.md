[![lint](https://github.com/agency-fund/xngin/actions/workflows/lint.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/lint.yaml)
[![precommit](https://github.com/agency-fund/xngin/actions/workflows/precommit.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/precommit.yaml)
[![test](https://github.com/agency-fund/xngin/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/test.yaml)

# Evidential<a name="evidential"></a>

<!-- mdformat-toc start --slug=github --maxlevel=2 --minlevel=1 -->

- [Evidential](#evidential)
  - [Getting Started](#getting-started)
  - [Documentation for Evidential](#documentation-for-evidential)
  - [Documentation on our Dependencies](#documentation-on-our-dependencies)
  - [Settings](#settings)
  - [API Keys](#api-keys)

<!-- mdformat-toc end -->

Evidential is an experiments management platform built on FastAPI, Postgres, and React.

## Getting Started<a name="getting-started"></a>

Follow the steps below to get a local development environment running.

1. Install [Task](https://taskfile.dev/).

1. Install dependencies (Atlas, uv, Python dependencies) by running:

   ```shell
   task install-dependencies
   ```

1. Install Docker.

1. Run the unit tests:

   ```shell
   # This creates a local Postgres instance on port 5499.
   task bootstrap-app
   # This runs the unit tests.
   task test
   ```

1. Obtain the OIDC development client secret from a teammate and add it to your .env file:

   ```shell
   echo GOOGLE_OIDC_CLIENT_SECRET=... >> .env
   ```

1. Get familiar with the task runner. Most of the commands you will run are defined in Taskfile.yml and using it helps
   ensure all developers are running in consistent environments. Run:

   ```shell
   task --list
   ```

1. Then start the dev server:

   ```shell
   task start
   ```

   This will start the server at http://localhost:8000. It stores its state in a local Postgres instance, running in
   Docker, on localhost:5499.

   > Note: Our system uses Google logins for SSO. If you do not have an @agency.fund Google login, run the following
   > command to allow yourself access to the UI:
   >
   > ```shell
   > task create-user
   > ```

1. If you are only working on the UI, you can skip the rest of the instructions.

1. Send some test requests:

   Each request should have an HTTP request header `Datasource-ID` set to the `id` value of a configuration entry in the
   [xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) settings file. For testing
   and local dev, use `testing`.

   ```shell
   curl -H "Datasource-ID: testing" 'http://localhost:8000/v1/filters?participant_type=test_participant_type'
   curl -H "Datasource-ID: testing" 'http://localhost:8000/v1/strata?participant_type=test_participant_type'
   ```

   See [TESTING.md](docs/TESTING.md) for instructions on how we use API test scripts.

1. Familiarize yourself with the xngin-cli:

```shell
   uv run xngin-cli --help
```

2. Visit the local interactive docs page: http://localhost:8000/docs

1. Now set up the pre-commit hooks in your local git with:

   ```shell
   uv run pre-commit install
   ```

   This will install git hooks to run formatters and linters whenever you create or modify a commit. It only does checks
   against files that changed; to force it to check all files, you can run:

   ```
   uv run pre-commit run -a
   ```

1. (deprecated) To parse a remote Google Sheets config, you'll need the service account credentials JSON blob. Copy it
   to `~/.config/gspread/service_account.json`.

   - [Setup a service account in your GCP console](console.cloud.google.com) > select your project via dropdown at the
     top > IAM & Admin > Service Accounts > + Create Service Account > give it a name, desc and create; *note the email
     addr created* > After creation, click the email for that account > Keys tab > Create the json key file and put it
     in the above location. Lastly, share the spreadsheet as Viewer-only with this special service account email
     address.
   - Ensure that the [Google Sheets API](https://console.developers.google.com/apis/api/sheets.googleapis.com/overview)
     is enabled for your Google Cloud project.

## Documentation for Evidential<a name="documentation-for-evidential"></a>

See [docs/](docs/) for detailed more detailed documentation.

## Documentation on our Dependencies<a name="documentation-on-our-dependencies"></a>

We use many advanced features of Pydantic, FastAPI, and SQLAlchemy. Before digging into the code, familiarize yourself
with these concepts:

- Pydantic:
  - [Pydantic models](https://docs.pydantic.dev/2.8/concepts/models/)
  - [Custom validation](https://docs.pydantic.dev/2.8/concepts/validators/)
  - [Serialization](https://docs.pydantic.dev/2.8/concepts/serialization/)
  - [Annotated Type](https://docs.pydantic.dev/2.8/concepts/fields/#using-annotated)
  - [Composing Types](https://docs.pydantic.dev/2.8/concepts/types/#composing-types-via-annotated)
- FastAPI:
  - [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/first-steps/)
  - [Adding metadata using `Annotated`](https://fastapi.tiangolo.com/python-types/#type-hints-with-metadata-annotations)
  - [Dependency Injection](https://fastapi.tiangolo.com/tutorial/dependencies/)
- OpenAPI:
  - The metadata annotations on FastAPI handlers and Pydantic type produce an OpenAPI specification document which we
    share with customers and use to generate a client library for the frontend.
  - [Generated API Documentation](https://main.dev.agencyfund.org/docs)
- [SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/tutorial/index.html):
  - We use SQLAlchemy to manage our application database and connect to customer data warehouses.
- Other dev tooling:
  - [Atlas](https://atlasgo.io/) for schema migrations.
  - [ruff](https://github.com/astral-sh/ruff) for linting and formatting (configured with [pyproject.toml](pyproject.toml)
  - [pre-commit](https://pre-commit.com/) for invoking automatic checks (configured with [.pre-commit-config.yaml](.pre-commit-config.yaml).
- To understand the frontend, see https://github.com/agency-fund/xngin-dash.

## Settings<a name="settings"></a>

> Note: This section needs updates.

Evidential reads settings from two places:

1. Static: A static file specified by the `XNGIN_SETTINGS` environment variable.
1. Database: A database.

Both sources are used by a running server. This allows you to have some settings that are configured via a static file,
and some that are read from the database.

| Settings Source | Supports Unauthenticated Datasources | Used in Testing | Data Type                                                                   |
| --------------- | ------------------------------------ | --------------- | --------------------------------------------------------------------------- |
| Static          | Yes                                  | Yes             | [`xngin.apiserver.settings:XnginSettings`](src/xngin/apiserver/settings.py) |
| Database        | No                                   | Generally not   | See [models](rc/xngin/apiserver/models/tables.py)                           |

### Static Settings

Values are read at startup from environment variables and from a JSON file specified by the `XNGIN_SETTINGS` environment
variable. This defaults to [xngin.settings.json](xngin.settings.json), but you will use
[xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) file during most of your work.

The settings files can be committed to version control but secret values should not be. They should be referred to with
`${secret:NAME}` syntax. When the `XNGIN_SECRETS_SOURCE` environment variable is unset or set to "environ", those
references will be replaced with a corresponding environment variable value.

Static settings define per-client configuration
allowing us to provide a multi-tenant service. It is retrieved in production via dependency injection (see
[`settings.py:get_settings_for_server`](src/xngin/apiserver/settings.py)), which is overridden in tests (see
[`conftest.py:get_settings_for_test`](src/xngin/apiserver/conftest.py)).

### What's in the settings?

Settings define **client-level** configuration whose schema is [
`ClientConfig`](src/xngin/apiserver/settings.py), which
can be one of a fixed set of supported customer configurations (e.g. `RemoteDatabaseConfig`) that package up how to
connect to the data warehouse (see `BaseDsn` and descendants) along with all the different `Participant` types (aka
each type of unit of experimentation, e.g. a WhatsApp group, or individual phone numbers, hospitals, schools, ...).

Within a datasource, we also store **Participant type-level** configuration with schema
[`inspection_types.py:ParticipantsSchema`](src/xngin/apiserver/dwh/inspection_types.py), including column and type info derived from
the warehouse via introspection (see
[`config_sheet.py:create_schema_from_table`](src/xngin/sheets/config_sheet.py),
[`main.py:get_sqlalchemy_table_from_engine`](src/xngin/apiserver/main.py)), as well as extra metadata about columns
(is_strata/is_filter/is_metric). The extra metadata may come from CSV or in Google spreadsheets as filled out by the
client. Both sources (dwh introspection, gsheets) are represented by the `ParticipantsSchema` model, although not all
information may be supplied by either.

## API Keys<a name="api-keys"></a>

The datasources defined in static settings (JSON) default to being unprotected. To require API keys for a specific
datasource, set the `require_api_key` flag to true on the settings. Example:

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
  'http://localhost:8000/v1/filters?participant_type=test_participant_type'
```
