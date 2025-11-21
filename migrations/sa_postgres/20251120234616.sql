-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "impact" character varying NOT NULL DEFAULT '', ADD COLUMN "decision" character varying NOT NULL DEFAULT '';
