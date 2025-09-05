-- Copy values from "context_val" to "context_vals" if needed
ALTER TABLE "public"."draws" ADD COLUMN "context_vals" double precision[] NULL;
UPDATE "public"."draws" SET "context_vals" = "context_val"
WHERE "context_val" IS NOT NULL;
-- Modify "draws" table
ALTER TABLE "public"."draws" DROP COLUMN "observation_type", DROP COLUMN "context_val";
