[![lint](https://github.com/agency-fund/xngin/actions/workflows/lint.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/lint.yaml)
[![precommit](https://github.com/agency-fund/xngin/actions/workflows/precommit.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/precommit.yaml)
[![test](https://github.com/agency-fund/xngin/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/agency-fund/xngin/actions/workflows/test.yaml)

# Evidential<a name="evidential"></a>

<!-- mdformat-toc start --slug=github --maxlevel=2 --minlevel=1 -->

- [Evidential](#evidential)
  - [Getting Started](#getting-started)
  - [Documentation](#documentation)

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

## Documentation<a name="documentation"></a>

See [docs/](docs/) for more documentation.
