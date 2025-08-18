-- Rename a column from "context_val" to "context_vals"
ALTER TABLE "public"."draws" ADD COLUMN "context_vals" double precision[] NULL;
UPDATE "public"."draws" SET "context_vals" = "context_val" WHERE "context_val" IS NOT NULL;
UPDATE "public"."draws" SET "context_val" = NULL;
ALTER TABLE "public"."draws" DROP COLUMN "context_val";