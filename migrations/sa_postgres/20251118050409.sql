-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "arm_weights" double precision[] NULL;
