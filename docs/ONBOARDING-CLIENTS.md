# Onboarding Clients<a name="onboarding-clients"></a>

<!-- mdformat-toc start --slug=github --maxlevel=2 --minlevel=1 -->

- [Onboarding Clients](#onboarding-clients)
  - [Onboarding new Clients](#onboarding-new-clients)
  - [Supported DWHs and DSN url format](#supported-dwhs-and-dsn-url-format)

<!-- mdformat-toc end -->

> Note: This process can now be completed via the xngin-dash UI. The JSON files are only useful if you want to create
> a configuration that doesn't require authentication.

## Onboarding new Clients<a name="onboarding-new-clients"></a>

1. Get credentials to the client's data warehouse that has at least read-only access to the schemas/datasets
   containing the table(s) of interest. Each table will be a different "participant type" the user wishes to experiment
   over, and should contain a) a unique id column, b) features to filter the partipcants with (i.e. target for
   experiment
   eligibility), and c) metrics to use as possible outcomes to track, and optionally d) features to stratify on.

1. Generate the participant-level column metadata. This will ultimately be a google sheet that we as the service
   provider own, but we share with the user to configure.

   1. First bootstrap column names and types from the dwh schema. There will be one row output per column in the target
      dwh table. See the command `uv run xngin-cli bootstrap-spreadsheet --help`
      1. If output as csv, import it to a new google spreadsheet that _we create and own_.
      1. Share it with our gsheet service account.
   1. Share it with the client to mark which columns are filters/metrics/strata and which to use as the unique_id.
   1. Additional table columns can be added (or removed) from the spreadsheet by the client.

1. Generate the client's config block in `xngin.settings.json`. Give them a unique string `"id:"` that they will pass
   back to us with every API request, specify `"type: "remote"` as the general type of dwh (see `settings.py`), provide
   dwh
   connectivity info in `"dwh":`, and lastly create the `"participants:"` list. Each item is a Participant object with a
   `"participant_type":` identifier for use in API requests, a `"table_name"` to look up in their dwh, and the GSheets
   URL
   and worksheet tab name from above to find its associated column metadata.

1. Bootstrap a set of data in the warehouse to use as a new Participant type.

For more examples, see the `xngin.gha.settings.json` settings used for testing.

## Supported DWHs and DSN url format<a name="supported-dwhs-and-dsn-url-format"></a>

- Redshift - `postgresql+psycopg2://username@host:port/databasename`
- Postgres - `postgresql+psycopg://username@host:port/databasename`
- BigQuery - `bigquery://some-project/some-dataset`

### BigQuery as the Customer's DWH Support<a name="bigquery-as-the-customers-dwh-support"></a>

BigQuery support is implemented but has not yet been fully tested.
See [.github/workflows/test.yaml](.github/workflows/test.yaml) for lifecycle tests.

#### Authentication<a name="authentication"></a>

- To authenticate with the customer's bigquery, only service account authentication is supported.

- _The customer_ should create a service account for us to access their warehouse with
  `BigQuery User` permissions to the client's Bigquery project, otherwise the server will get the
  `User does not have bigquery.jobs.create permission in project <project_name>` error.

- All interactions with the customer's warehouse happen via the explicitly configured
  authentication in the _settings.json_ files, which should correspond to the service account noted
  above. See [xngin.gha.settings.json](xngin.gha.settings.json) for an example.

- ⚠️ _As the service provider_, we create and own the initial customer warehouse configuration
  spreadsheets for the customer and share access to them for further modification. All interactions
  with these spreadsheets happen with the
  environment variable `GSHEET_GOOGLE_APPLICATION_CREDENTIALS`, which should point to _our_ service account
  that has access to the sheets we create. Do not confuse these credentials with the customers' service account in
  settings.json!

  Example:

  ```shell
  GSHEET_GOOGLE_APPLICATION_CREDENTIALS=secrets/customer_service_account.json \
    xngin-cli bootstrap-spreadsheet \
    bigquery://project/dataset res_users --unique-id-col user_id
  ```
