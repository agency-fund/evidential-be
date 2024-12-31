data "external_schema" "sqlalchemy_sqlite" {
  program = [
    "atlas-provider-sqlalchemy",
    "--path", "./src/xngin/apiserver/models",
    "--dialect", "sqlite"
  ]
}

data "external_schema" "sqlalchemy_pg" {
  program = [
    "atlas-provider-sqlalchemy",
    "--path", "./src/xngin/apiserver/models",
    "--dialect", "postgresql"
  ]
}

env "sa_sqlite" {
  src = data.external_schema.sqlalchemy_sqlite.url
  dev = "sqlite://dev?mode=memory"
  migration {
    dir = "file://migrations/sa_sqlite"
  }
  format {
    migrate {
      diff = "{{ sql . \"  \" }}"
    }
  }
}

env "sa_postgres" {
  src = data.external_schema.sqlalchemy_pg.url
  // Railway is running on Postgres 16.
  dev = "docker://postgres/16/railway"
  migration {
    dir = "file://migrations/sa_postgres"
  }
  format {
    migrate {
      diff = "{{ sql . \"  \" }}"
    }
  }
}
