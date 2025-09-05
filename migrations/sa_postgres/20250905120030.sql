-- Modify "draws" table
ALTER TABLE "public"."draws" DROP COLUMN "observation_type", DROP COLUMN "context_val", ADD COLUMN "context_vals" double precision[] NULL;
