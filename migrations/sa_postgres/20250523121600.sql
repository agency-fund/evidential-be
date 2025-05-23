-- Modify "experiments" table
ALTER TABLE "public"."experiments" ALTER COLUMN "state" TYPE character varying USING "state"::text;
-- Drop enum type "experimentstate"
DROP TYPE "public"."experimentstate";

-- Custom migration for the old enum values: convert upper case to lower.
UPDATE "public"."experiments" SET "state" = LOWER("state");