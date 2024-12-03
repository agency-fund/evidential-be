# Defines a production runtime environment for the service. This is not for development uses.
FROM python:3.12
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /bin/uv
WORKDIR /code
COPY pyproject.toml /code/
COPY uv.lock /code/
RUN --mount=type=cache,target=/root/.cache/uv /bin/uv sync --frozen --no-install-project
COPY xngin.settings.json .
COPY ./src /code/src
RUN --mount=type=cache,target=/root/.cache/uv /bin/uv sync --frozen
CMD ["uv", "run", "fastapi", "run", "src/xngin/apiserver/main.py"]
