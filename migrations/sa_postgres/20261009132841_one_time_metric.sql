-- Modify "experiment_fields" table
ALTER TABLE "public"."experiment_fields" ADD COLUMN "use_one_time_metric" boolean NOT NULL DEFAULT false;
