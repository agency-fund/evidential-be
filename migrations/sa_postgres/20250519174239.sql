-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "power" double precision NULL, ADD COLUMN "alpha" double precision NULL, ADD COLUMN "fstat_thresh" double precision NULL;
