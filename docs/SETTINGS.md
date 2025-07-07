# Static Settings<a name="static-settings"></a>

<!-- mdformat-toc start --slug=github --maxlevel=2 --minlevel=1 -->

- [Static Settings](#static-settings)
  - [Background](#background)
  - [Static Settings](#static-settings-1)
  - [What's in the settings?](#whats-in-the-settings)
  - [API Keys](#api-keys)

<!-- mdformat-toc end -->

## Background<a name="background"></a>

Evidential reads settings from two places:

1. Static (optional): A static file specified by the `XNGIN_SETTINGS` environment variable.
1. Database: The Postgres database.

Both sources are used by a running server. This allows you to have some settings that are configured via a static file,
and some that are read from the database.

| Settings Source | Supports Unauthenticated Datasources | Used in Testing | Available via Experiments API | Available via Admin API | Data Type                                                                   |
| --------------- | ------------------------------------ | --------------- | ----------------------------- | ----------------------- | --------------------------------------------------------------------------- |
| Static          | Yes                                  | Yes             | Yes                           | No                      | [`xngin.apiserver.settings:XnginSettings`](src/xngin/apiserver/settings.py) |
| Database        | No                                   | Yes             | Yes                           | Yes                     | See [models](rc/xngin/apiserver/models/tables.py)                           |

## Static Settings<a name="static-settings-1"></a>

Static settings are generally only used in deployment scenarios where the UI is not being used to manage datasources or
experiments.

Static settings are read at startup from environment variables and from a JSON file specified by the `XNGIN_SETTINGS`
environment variable.
[xngin.testing.settings.json](src/xngin/apiserver/testdata/xngin.testing.settings.json) is used in tests.

The settings files can be committed to version control but secret values should not be. They should be referred to with
`${secret:NAME}` syntax. When the `XNGIN_SECRETS_SOURCE` environment variable is unset or set to "environ", those
references will be replaced with a corresponding environment variable value.

Static settings define per-client configuration
allowing us to provide a multi-tenant service. It is retrieved in production via dependency injection (see
[`settings.py:get_settings_for_server`](src/xngin/apiserver/settings.py)), which is overridden in tests (see
[`conftest.py:get_settings_for_test`](src/xngin/apiserver/conftest.py)).

## What's in the settings?<a name="whats-in-the-settings"></a>

Settings define **client-level** configuration whose schema is [
`ClientConfig`](src/xngin/apiserver/settings.py), which
can be one of a fixed set of supported customer configurations (e.g. `RemoteDatabaseConfig`) that package up how to
connect to the data warehouse (see `BaseDsn` and descendants) along with all the different `Participant` types (aka
each type of unit of experimentation, e.g. a WhatsApp group, or individual phone numbers, hospitals, schools, ...).

Within a datasource, we also store **Participant type-level** configuration with schema
[`inspection_types.py:ParticipantsSchema`](src/xngin/apiserver/dwh/inspection_types.py), including column and type info
derived from
the warehouse via introspection (see
[`config_sheet.py:create_schema_from_table`](src/xngin/sheets/config_sheet.py),
[`main.py:get_sqlalchemy_table_from_engine`](src/xngin/apiserver/main.py)), as well as extra metadata about columns
(is_strata/is_filter/is_metric). The extra metadata may come from CSV or in Google spreadsheets as filled out by the
client. Both sources (dwh introspection, gsheets) are represented by the `ParticipantsSchema` model, although not all
information may be supplied by either.

## API Keys<a name="api-keys"></a>

Datasources defined in the database always require an API key to interact with.

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
