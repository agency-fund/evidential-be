# Defines a production runtime environment for the service. This is not for development uses.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_LINK_MODE=copy
WORKDIR /code
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,ro \
    --mount=type=bind,source=uv.lock,target=uv.lock,ro \
    /usr/local/bin/uv sync --locked --no-install-project --no-dev
COPY pyproject.toml .
COPY uv.lock .
COPY ./src /code/src
RUN --mount=type=cache,target=/root/.cache/uv \
    /usr/local/bin/uv sync --locked --no-dev
CMD ["./.venv/bin/xngin-apiserver-live"]
