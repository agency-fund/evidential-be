-- BigQuery has no UUID type; uuid_filter is a STRING. DECIMAL maps to BIGNUMERIC because the CSV carries more than the
-- 9 fractional digits NUMERIC allows (both are Numeric to SQLAlchemy and fudo). TIMESTAMP (naive) maps to DATETIME and
-- TIMESTAMPTZ to TIMESTAMP, which BigQuery stores as an instant in UTC.
CREATE TABLE {{table_name}} (
  gender STRING,
  ethnicity STRING,
  first_name STRING,
  last_name STRING,
  id INT64 NOT NULL,
  income BIGNUMERIC,
  potential_0 BIGNUMERIC,
  potential_1 INT64,
  is_recruited BOOL,
  is_registered BOOL,
  is_onboarded BOOL,
  is_onboarded_onetime BOOL,
  is_engaged BOOL,
  is_retained BOOL,
  baseline_income BIGNUMERIC,
  current_income BIGNUMERIC,
  uuid_filter STRING,
  sample_date DATE,
  sample_timestamp DATETIME,
  timestamp_with_tz TIMESTAMP NOT NULL
);
