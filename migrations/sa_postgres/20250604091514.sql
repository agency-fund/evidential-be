-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "stopped_assignments_at" timestamptz NULL, ADD COLUMN "stopped_assignments_reason" character varying NULL;
