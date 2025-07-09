# Evidential & Docker<a name="evidential--docker"></a>

<!-- mdformat-toc start --slug=github --maxlevel=2 --minlevel=1 -->

- [Evidential & Docker](#evidential--docker)
  - [How Evidential Uses Docker](#how-evidential-uses-docker)
  - [Automated Docker Image Builds](#automated-docker-image-builds)
  - [How do I run the Docker images built by the CI?](#how-do-i-run-the-docker-images-built-by-the-ci)
  - [Running the Docker container locally](#running-the-docker-container-locally)

<!-- mdformat-toc end -->

> Note: In development environments, we do not use Docker to run the backend or frontend. But we optionally use Docker
> for running a local Postgres server. If you're only developing locally, ignore this document.

## How Evidential Uses Docker<a name="how-evidential-uses-docker"></a>

We provide a [basic Dockerfile](../Dockerfile) that can be customized for various customer deployment environments. We
also have one that [works on Railway](../Dockerfile.railway). These images are intended for **production**. In local development
scenarios, you'll be running directly in your source repository without any containers (except for PostgreSQL, which is
optional).

## Automated Docker Image Builds<a name="automated-docker-image-builds"></a>

Docker Images are created automatically by CI whenever a commit is merged into the main branch. These are intended to
be used by customers for deployments and to validate that dependencies install correctly.

## How do I run the Docker images [built by the CI](https://github.com/agency-fund/evidential-be/pkgs/container/xngin)?<a name="how-do-i-run-the-docker-images-built-by-the-ci"></a>

```shell
# Authenticate your local Docker to the ghcr using a GitHub Personal Access Token (classic) that has at least the
# `read:packages` scope. Create one here: https://github.com/settings/tokens
docker login ghcr.io -u YOUR_GITHUB_USERNAME
```

Then you can start the container locally with something like this:

```shell
docker run \
  -it \
  --env-file ./.env \
  -v path/to/settings/directory:/settings \
  -e XNGIN_SETTINGS=/settings/xngin.settings.json \
  -p 127.0.0.1:8000:8000 \
  ghcr.io/agency-fund/xngin:main
```

This also means we can fetch prod settings from elsewhere on our host and mount the dir with it in the container.
Don't forget to add other environment variables not set on the command line to a `.env` file such as
`GSHEET_GOOGLE_APPLICATION_CREDENTIALS`; these are not part of the image. Also beware that Docker
[includes double quotes in the value of your env variables](https://github.com/docker/compose/issues/3702)
read in via --env-file.

## Running the Docker container locally<a name="running-the-docker-container-locally"></a>

You can also build the container locally (without relying on the CI pipeline).

```shell
docker build -t xngin .
docker run \
  -it \
  --env-file secrets/.env \
  -v `pwd`/settings/:/settings \
  -e XNGIN_SETTINGS=/settings/xngin.settings.json \
  -e GSHEET_GOOGLE_APPLICATION_CREDENTIALS=/settings/gs_service_account.json \
  -p 8000:8000  \
   xngin:latest
```

See [.github/workflows/test.yaml](.github/workflows/test.yaml) for an example of how to run tests locally within the
Docker container.
