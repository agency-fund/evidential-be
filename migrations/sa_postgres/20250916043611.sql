-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "design_url" character varying NOT NULL DEFAULT '';
