-- Modify "experiment_fields" table
ALTER TABLE "public"."experiment_fields" ADD COLUMN "is_filter" boolean NOT NULL DEFAULT false;
