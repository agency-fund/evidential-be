# testdata

Also see:

- pytest fixtures that use data in this directory: [conftest](../../../../src/xngin/apiserver/conftest.py)
- [Updating Testing DWH](../../../../docs/UPDATING-TESTING-DWH.md)

## xngin.testing.settings.json (deprecated)

This file contains a number of statically configured `settings.py::Datasource` objects as json for
use in testing. All ds configs are currently of type `RemoteDatabaseConfig` as we removed the local
SQLite-as-dwh support a while back.

### Datasource-ID: testing

| Filename            | Description                                    |
| ------------------- | ---------------------------------------------- |
| testing_dwh.csv.zst | zstd-compressed dump of a fake data warehouse. |

Tests create a testing_dwh.db database as needed in this directory from testing_dwh.csv.zst.

### Datasource-ID: testing

The test list of `participants` has a single ParticipantsConfig of the `ParticipantsDef` type (i.e.
type = "schema"). This is used to test having the schema info inlined directly into the containing
Datasource's config (which is also how we represent the ds schema when read directly from the dwh
and cached in our own app db).
