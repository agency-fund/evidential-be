import psycopg as psycopg3


def create_database_if_not_exists_pg(connect_url):
    tmpl_url = connect_url.set(database="postgres")
    try:
        with psycopg3.connect(
            tmpl_url.render_as_string(hide_password=False), autocommit=True
        ) as conn:
            conn.execute(f"CREATE DATABASE {connect_url.database}")
            print(f"\n✨ Created database '{connect_url.database}'")
    except psycopg3.errors.DuplicateDatabase:
        print(f"📊 Database '{connect_url.database}' already exists")
