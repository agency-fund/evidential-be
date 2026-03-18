-- Modify "experiment_filters" table
ALTER TABLE "public"."experiment_filters" ADD COLUMN "boolean_values" boolean[] NULL;
