import psycopg as psycopg3
import sqlalchemy


def create_database_if_not_exists_pg(connect_url: sqlalchemy.URL | str):
    # Derive a psycopg3-compatible Postgres DSN from the SQLAlchemy URL by stripping the drivername.
    # Also, connect to the postgres database because we know it exists already.
    if isinstance(connect_url, str):
        connect_url = sqlalchemy.engine.url.make_url(connect_url)
    if connect_url.get_backend_name() != "postgresql":
        print(f"⚠️  No database created: provided URL backend is '{connect_url.get_backend_name()}', not 'postgresql'.")
        return

    tmpl_url = connect_url.set(database="postgres", drivername="postgres")
    psycopg3_compatible = tmpl_url.render_as_string(hide_password=False)
    try:
        with psycopg3.connect(psycopg3_compatible, autocommit=True) as conn:
            conn.execute(f"CREATE DATABASE {connect_url.database}")
            print(f"\n✨ Created database '{connect_url.database}'")
    except psycopg3.errors.DuplicateDatabase:
        print(f"📊 Database '{connect_url.database}' already exists")
