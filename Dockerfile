# Defines a production runtime environment for the service. This is not for development uses.
FROM python:3.12
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
COPY --from=ghcr.io/astral-sh/uv:0.2.37 /uv /bin/uv
WORKDIR /code
COPY ./requirements.linux_x86_64.txt /code/requirements.txt
# These command line flags restrict the installation *only* to Python dependencies pre-specified in the
# requirements.txt file. This helps ensure we have a more-deterministic runtime environment between builds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
      --no-deps \
      --require-hashes \
      --only-binary :all: \
      --no-binary psycopg2 \
      -r /code/requirements.txt
COPY ./app /code/app
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
