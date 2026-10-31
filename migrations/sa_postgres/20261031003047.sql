-- Drop index "idx_tasks_embargo" from table: "tasks"
DROP INDEX "public"."idx_tasks_embargo";
-- Create index "idx_tasks_active" to table: "tasks"
CREATE INDEX "idx_tasks_active" ON "public"."tasks" ("created_at") WHERE (((status)::text = 'pending'::text) OR ((status)::text = 'running'::text));
