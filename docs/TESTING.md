# Testing<a name="testing"></a>

<!-- mdformat-toc start --slug=github --maxlevel=2 --minlevel=1 -->

- [Testing](#testing)
  - [Unit Tests](#unit-tests)
  - [Google Integration Tests](#google-integration-tests)
  - [BigQuery Integration Tests](#bigquery-integration-tests)
  - [Showing additional test output](#showing-additional-test-output)
  - [API Test Scripts](#api-test-scripts)
  - [Loading the testing DWH with a different schema?](#loading-the-testing-dwh-with-a-different-schema)
  - [Testing environment for BigQuery as the Service Provider](#testing-environment-for-bigquery-as-the-service-provider)

<!-- mdformat-toc end -->

This is a collection of random tips about testing. Most of the test configurations you will use are already defined
in the [Taskfile](../Taskfile.yml).

> Warning: If you find yourself doing any testing that doesn't have `task` entry for it already, let's talk!

## Unit Tests<a name="unit-tests"></a>

Run unit tests with:

```shell
task test
```

We run unittests with [pytest](https://docs.pytest.org/en/stable/).

The `task test` helper automatically creates a local Postgres instance for testing and creates a testing datawarehouse
in the "dwh" database from [testing_dwh.csv.zst](../src/xngin/apiserver/testdata/testing_dwh.csv.zst) file.
[testing_sheet.csv](../src/xngin/apiserver/testdata/testing_sheet.csv) is the corresponding spreadsheet that simulates a
typical table configuration for the participant type data above.

[Various tests](../.github/workflows/test.yaml) are also run as part of our GitHub action test workflow.

[conftest.py](../src/xngin/apiserver/conftest.py) defines fixtures used by many of the tests.

## Google Integration Tests<a name="google-integration-tests"></a>

Some of our pytests have a test marked as 'integration'. These are only usually run in GHA but you can run them locally
by setting `GOOGLE_APPLICATION_CREDENTIALS` and running:

```shell
uv run pytest -m integration
```

## BigQuery Integration Tests<a name="bigquery-integration-tests"></a>

Method 1:

1. Push the branch to GitHub (i.e. draft PR)
1. Use the GitHub CLI to trigger the BigQuery integration tests on a specific branch:

```shell
gh workflow run tests --ref your-branch-name -f run-bq-integration=true
```

`tests` refers to the name of the workflow containing the integration tests. `run-bq-integration` refers to the name
of the workflow_dispatch input that determines if the BigQuery integration tests are run. The branch must already be
pushed to GitHub.

Method 2: Trigger via GitHub UI

1. Push the branch to GitHub (i.e. draft PR)
1. Click https://github.com/agency-fund/xngin/actions/workflows/test.yaml
1. Click "new workflow"
1. Select the PR's branch name
1. Tick the "bq integration" box.

You can also trigger the BigQuery integration tests to run in GHA by putting `run:bqintegration` in your PR comment.

## Showing additional test output<a name="showing-additional-test-output"></a>

`pytest -rA` to print out _all_ stdout from your tests; `-rx` for just those failing. (See
[docs](https://docs.pytest.org/en/latest/how-to/output.html#producing-a-detailed-summary-report) for more info.)

## API Test Scripts<a name="api-test-scripts"></a>

See [apitest.strata.xurl](../src/xngin/apiserver/routers/stateless/testdata/apitest.strata.xurl) for a complete example
of how to write an
API test script. We use a small custom
file format called [Xurl](../src/xngin/apiserver/testing/xurl.py).

`test_api.py` tests that use `testdata/*.xurl` data can be automatically updated with the actual server results by
prefixing your pytest run with the environment variable: `UPDATE_API_TESTS=1`.

Or just run `task update-api-tests`.

## Loading the testing DWH with a different schema?<a name="loading-the-testing-dwh-with-a-different-schema"></a>

Use the CLI command:

```shell
uv run xngin-cli create-testing-dwh \
   --dsn=$DATABASE_URL \
   --password=$PASSWORD \
   --table-name=test_participant_type \
   --schema-name=alt
```

### How can I run the unittests against an arbitrary data warehouse?<a name="how-can-i-run-the-unittests-that-use-my-pgbq-instance-as-the-test-dwh"></a>

Tests that exercise integration with a data warehouse do so via two mechanisms:

1. xngin.testing.settings.json defines some static configuration for stable databases.
1. The XNGIN_QUERIES_TEST_URI environment variable defines a SQLAlchemy URI that some of the apiserver.dwh tests are run
   against.

By default, `task test` will run all the tests against Postgres. See the bq-integration GHA job for an example of how
to run some of the tests against BigQuery.

## Testing environment for BigQuery as the Service Provider<a name="testing-environment-for-bigquery-as-the-service-provider"></a>

You can create a test dataset on a project you've configured with bigquery
using our xngin-cli tool:

```shell
export GSHEET_GOOGLE_APPLICATION_CREDENTIALS=~/.config/gspread/service_account.json
xngin-cli create-testing-dwh --dsn 'bigquery://xngin-development-dc/ds'
```

The tool is also used for interacting with the BQ API directly
(i.e. `bigquery_dataset_set_default_expiration`).

These commands use Google's [Application Default Credentials]
(https://cloud.google.com/docs/authentication/application-default-credentials) process.

> Note: The GHA service account has permissions to access the xngin-development-dc.ds dataset.

> Note: You do not need the gcloud CLI or related tooling installed.
