-- Modify "experiment_fields" table
ALTER TABLE "public"."experiment_fields" DROP CONSTRAINT "experiment_fields_pkey", DROP COLUMN "id", DROP COLUMN "use", DROP COLUMN "other", ADD COLUMN "is_unique_id" boolean NOT NULL DEFAULT false, ADD COLUMN "is_strata" boolean NOT NULL DEFAULT false, ADD COLUMN "is_primary_metric" boolean NOT NULL DEFAULT false, ADD COLUMN "metric_pct_change" double precision NULL, ADD COLUMN "metric_target" double precision NULL, ADD PRIMARY KEY ("experiment_id", "field_name");
-- Create "experiment_filters" table
CREATE TABLE "public"."experiment_filters" (
  "id" character varying NOT NULL,
  "experiment_id" character varying(36) NOT NULL,
  "field_name" character varying(255) NOT NULL,
  "relation" character varying(20) NOT NULL,
  "string_values" character varying(255)[] NULL,
  "numeric_values" numeric[] NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "experiment_filters_experiment_id_field_name_fkey" FOREIGN KEY ("experiment_id", "field_name") REFERENCES "public"."experiment_fields" ("experiment_id", "field_name") ON UPDATE NO ACTION ON DELETE CASCADE
);
