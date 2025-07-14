-- Modify "context" table
ALTER TABLE "public"."context" ALTER COLUMN "description" SET NOT NULL;
-- Modify "draws" table
ALTER TABLE "public"."draws" ALTER COLUMN "context_val" DROP NOT NULL;
