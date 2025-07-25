[![lint](https://github.com/agency-fund/evidential-be/actions/workflows/lint.yaml/badge.svg?branch=main)](https://github.com/agency-fund/evidential-be/actions/workflows/lint.yaml)
[![test](https://github.com/agency-fund/evidential-be/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/agency-fund/evidential-be/actions/workflows/test.yaml)

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

1. Install Docker.

1. Install dependencies (Atlas, uv, git-lfs, Python dependencies) by running:

   ```shell
   task install-dependencies
   ```

1. Run the unit tests:

   ```shell
   task test
   ```

1. Obtain the OIDC development client secret from a teammate and add it to your .env file:

   ```shell
   echo GOOGLE_OIDC_CLIENT_SECRET=... >> .env
   ```

1. Get familiar with the task runner. Most of the commands you will run are defined in Taskfile.yml. Run:

   ```shell
   task --list
   ```

1. Start the dev server:

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

1. Visit the local interactive OpenAPI docs page: http://localhost:8000/docs

1. Now set up the pre-commit hooks in your local git with:

   ```shell
   uv run pre-commit install
   ```

## Documentation<a name="documentation"></a>

See [https://docs.evidential.dev/](https://docs.evidential.dev/) for more documentation.
