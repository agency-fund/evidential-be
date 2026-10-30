-- Drop index "ix_draws_arm_id_created_at" from table: "draws"
DROP INDEX "public"."ix_draws_arm_id_created_at";
-- Drop index "ix_draws_pending" from table: "draws"
DROP INDEX "public"."ix_draws_pending";
-- Drop index "ix_draws_pending_autofail" from table: "draws"
DROP INDEX "public"."ix_draws_pending_autofail";
-- Modify "draws" table
ALTER TABLE "public"."draws" DROP COLUMN "enable_autofail";
-- Create index "ix_draws_pending" to table: "draws"
CREATE INDEX "ix_draws_pending" ON "public"."draws" ("experiment_id", "created_at") WHERE (outcome IS NULL);
-- Create index "ix_draws_resolved" to table: "draws"
CREATE INDEX "ix_draws_resolved" ON "public"."draws" ("arm_id", "observed_at" DESC) WHERE (outcome IS NOT NULL);
