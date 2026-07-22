-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "enable_autofail" boolean NOT NULL DEFAULT false, ADD COLUMN "autofail_window" integer NULL;
