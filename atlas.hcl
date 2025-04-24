data "external_schema" "sqlalchemy_pg" {
  program = [
    "atlas-provider-sqlalchemy",
    "--path", "./src/xngin/apiserver/models",
    "--dialect", "postgresql"
  ]
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

lint {
  non_linear {
    error = true
  }
}
