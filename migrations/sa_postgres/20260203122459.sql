-- Create "experiment_fields" table
CREATE TABLE "public"."experiment_fields" (
  "id" character varying NOT NULL,
  "experiment_id" character varying(36) NOT NULL,
  "field_name" character varying(255) NOT NULL,
  "use" character varying(20) NOT NULL,
  "data_type" character varying(50) NULL,
  "other" jsonb NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "experiment_fields_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Create index "idx_experiment_fields_experiment_id" to table: "experiment_fields"
CREATE INDEX "idx_experiment_fields_experiment_id" ON "public"."experiment_fields" ("experiment_id");
