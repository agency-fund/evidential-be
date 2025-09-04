-- Pre-migration check: Ensure observation_type is NULL everywhere
UPDATE "public"."draws" SET "observation_type" = NULL WHERE "observation_type" IS NOT NULL;
-- Modify "draws" table
ALTER TABLE "public"."draws" DROP COLUMN "observation_type";
