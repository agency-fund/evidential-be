# Defines a production runtime environment for the service. This is not for development uses.
FROM python:3.12
ENV PYTHONUNBUFFERED=1
COPY --from=ghcr.io/astral-sh/uv:0.3.1 /uv /bin/uv
WORKDIR /code
COPY pyproject.toml /code/
COPY uv.lock /code/
RUN --mount=type=cache,target=/root/.cache/uv /bin/uv sync --frozen
COPY ./app /code/app
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
