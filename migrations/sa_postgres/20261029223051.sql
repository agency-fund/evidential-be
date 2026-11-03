-- Create index "ix_draws_pending" to table: "draws"
CREATE INDEX "ix_draws_pending" ON "public"."draws" ("experiment_id") WHERE (outcome IS NULL);
