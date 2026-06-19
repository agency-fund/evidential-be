-- Modify "experiment_fields" table
ALTER TABLE "public"."experiment_fields" ADD COLUMN "is_target" boolean NOT NULL DEFAULT false;
