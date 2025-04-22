-- Modify "events" table
ALTER TABLE "public"."events" ALTER COLUMN "created_at" SET DEFAULT now();
