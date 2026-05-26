-- Modify "experiment_fields" table
ALTER TABLE "public"."experiment_fields" ADD COLUMN "is_cluster_key" boolean NOT NULL DEFAULT false;
