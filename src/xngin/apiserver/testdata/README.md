# testdata

This directory contains assets used in automated tests.

| **File**                    | **Purpose**                                                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| testing_dwh.\*.ddl          | Customizes the DDL used when loading the testing data warehouse into a database.                             |
| testing_dwh.csv.zst         | Testing DWH containing data used in tests and local development.                                             |
| xngin.gha.settings.json     | Defines various DatasourceConfig used by the continuous integration tests.                                   |
| xngin.testing.settings.json | Defines a single DatasourceConfig for use in tests. Inline schema corresponds to the testing data warehouse. |
