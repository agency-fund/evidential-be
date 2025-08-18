-- Rename a column from "context_val" to "context_vals"
ALTER TABLE "public"."draws" ADD COLUMN "context_vals" double precision[] NULL;
UPDATE "public"."draws" SET context_vals = context_val WHERE context_vals IS NULL;
ALTER TABLE "public"."draws" DROP COLUMN context_val;