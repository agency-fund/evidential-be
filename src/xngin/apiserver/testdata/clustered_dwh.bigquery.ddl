-- Clustered test data schema for cluster randomization testing (see clustered_dwh.postgres.ddl for the column notes).
-- BigQuery has no indexes; converted and engaged are stored as BOOL from the CSV's 0/1 values.
CREATE TABLE {{table_name}} (
  participant_id INT64 NOT NULL,
  cluster_equal INT64 NOT NULL,
  cluster_moderate INT64 NOT NULL,
  cluster_powerlaw INT64 NOT NULL,
  income BIGNUMERIC,
  test_score BIGNUMERIC,
  converted BOOL,
  engaged BOOL
);
