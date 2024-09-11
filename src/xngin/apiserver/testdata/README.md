# testdata

Also see: src/xngin/apiserver/testing/testdatasetup.py.

## config-id: customer-testing

Currently unused.

## config-id: testing

| Filename            | Description                                                                                                                                                                           |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| testing_dwh.csv.zst         | zstd-compressed dump of a fake data warehouse.                                                                                                                                        |
| dwh.configsheet.csv | configuration spreadsheet for that fake data warehouse. This is a CSV export of https://docs.google.com/spreadsheets/d/redacted/edit?gid=0#gid=0. |

Tests create a testing_dwh.db database as needed in this directory from testing_dwh.csv.zst.
