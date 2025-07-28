data "external_schema" "sqlalchemy_pg" {
  program = [
    "atlas-provider-sqlalchemy",
    "--path", "./src/xngin/apiserver/sqla",
    "--dialect", "postgresql"
  ]
}

env "sa_postgres" {
  src = data.external_schema.sqlalchemy_pg.url
  // Specify the version of Postgres that we are running in production.
  dev = "docker://postgres/17/railway"
  migration {
    dir = "file://migrations/sa_postgres"
  }
  format {
    migrate {
      diff = "{{ sql . \"  \" }}"
    }
  }
}

lint {
  non_linear {
    error = true
  }
}
