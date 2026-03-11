-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "datasource_table" character varying(255) NULL;
-- Create "experiment_fields" table
CREATE TABLE "public"."experiment_fields" (
  "experiment_id" character varying(36) NOT NULL,
  "field_name" character varying(255) NOT NULL,
  "data_type" character varying(50) NOT NULL,
  "is_unique_id" boolean NOT NULL DEFAULT false,
  "is_strata" boolean NOT NULL DEFAULT false,
  "is_primary_metric" boolean NOT NULL DEFAULT false,
  "metric_pct_change" double precision NULL,
  "metric_target" double precision NULL,
  PRIMARY KEY ("experiment_id", "field_name"),
  CONSTRAINT "experiment_fields_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Create "experiment_filters" table
CREATE TABLE "public"."experiment_filters" (
  "id" character varying NOT NULL,
  "position" integer NOT NULL,
  "experiment_id" character varying(36) NOT NULL,
  "field_name" character varying(255) NOT NULL,
  "relation" character varying(20) NOT NULL,
  "string_values" character varying(255)[] NULL,
  "numeric_values" numeric[] NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "experiment_filters_experiment_id_field_name_fkey" FOREIGN KEY ("experiment_id", "field_name") REFERENCES "public"."experiment_fields" ("experiment_id", "field_name") ON UPDATE NO ACTION ON DELETE CASCADE,
  CONSTRAINT "experiment_filters_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Create index "idx_experiment_filters_experiment_id_field_name" to table: "experiment_filters"
CREATE INDEX "idx_experiment_filters_experiment_id_field_name" ON "public"."experiment_filters" ("experiment_id", "field_name");
