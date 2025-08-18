-- Rename a column from "context_val" to "context_vals"
ALTER TABLE "public"."draws" ADD COLUMN "context_vals" double precision[] NULL;
UPDATE "public"."draws" SET "context_vals" = "context_val" WHERE "context_val" IS NOT NULL;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM "public"."draws" 
        WHERE "context_val" IS NOT NULL 
        AND ("context_vals" IS NULL OR "context_vals" = '{}')
        LIMIT 1
    ) THEN
        RAISE EXCEPTION 'Data migration incomplete: context_val contains data not migrated to context_vals';
    END IF;
END $$;

ALTER TABLE "public"."draws" DROP COLUMN "context_val";