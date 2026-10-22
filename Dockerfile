# Defines a production runtime environment for the service. This is not for development uses.

# The builder stage needs gcc, pg_config, and git to build and fetch dependencies (incl psycopg2).
FROM python:3.14-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0
WORKDIR /code
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,ro \
    --mount=type=bind,source=uv.lock,target=uv.lock,ro \
    /bin/uv sync --locked --no-install-project --no-dev
COPY pyproject.toml .
COPY uv.lock .
COPY ./src /code/src
RUN --mount=type=cache,target=/root/.cache/uv \
    /bin/uv sync --locked --no-dev

FROM python:3.14-slim-trixie
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1
WORKDIR /code
COPY --from=builder /code /code
CMD ["./.venv/bin/xngin-apiserver-live"]
