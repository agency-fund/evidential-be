-- Modify "draws" table
ALTER TABLE "public"."draws" ADD COLUMN "applied_at" timestamptz NULL;
-- Create index "ix_draws_unapplied" to table: "draws"
CREATE INDEX "ix_draws_unapplied" ON "public"."draws" ("arm_id") WHERE ((outcome IS NOT NULL) AND (applied_at IS NULL));
