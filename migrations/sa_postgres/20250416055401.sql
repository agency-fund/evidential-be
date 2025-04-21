-- Modify "tasks" table
ALTER TABLE "public"."tasks" ADD COLUMN "status" character varying NOT NULL DEFAULT 'pending';
-- Set comment to column: "status" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."status" IS 'Status of the task: ''pending'', ''running'', ''success'', or ''dead''.';
