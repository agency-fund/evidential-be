-- Modify "draws" table
ALTER TABLE "public"."draws" ADD COLUMN "enable_autofail" boolean NOT NULL DEFAULT false;
-- Create index "ix_draws_pending_autofail" to table: "draws"
CREATE INDEX "ix_draws_pending_autofail" ON "public"."draws" ("experiment_id", "created_at") WHERE ((enable_autofail IS TRUE) AND (outcome IS NULL));
