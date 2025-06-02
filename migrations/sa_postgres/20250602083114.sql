-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "stopped_at" timestamptz NULL, ADD COLUMN "stopped_reason" character varying NULL;
