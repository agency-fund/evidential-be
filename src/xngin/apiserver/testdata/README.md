# testdata

Also see:

- pytest fixtures that use data in this directory: [conftest](../../../../src/xngin/apiserver/conftest.py)
- [Updating Testing DWH](../../../../docs/UPDATING-TESTING-DWH.md)

## Datasource-ID: customer-test

Currently unused.

## Datasource-ID: testing

| Filename            | Description                                    |
| ------------------- | ---------------------------------------------- |
| testing_dwh.csv.zst | zstd-compressed dump of a fake data warehouse. |

Tests create a testing_dwh.db database as needed in this directory from testing_dwh.csv.zst.
