-- Modify "tasks" table
ALTER TABLE "public"."tasks" ADD COLUMN "message" character varying NULL;
-- Set comment to column: "message" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."message" IS 'An optional informative message about the state of this task.';
