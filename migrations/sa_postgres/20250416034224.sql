-- Create "tasks" table
CREATE TABLE "public"."tasks" (
  "id" character varying NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  "task_type" character varying NOT NULL,
  "embargo_until" timestamptz NULL,
  "retry_count" integer NOT NULL,
  "payload" jsonb NULL,
  "event_id" character varying NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "tasks_event_id_fkey" FOREIGN KEY ("event_id") REFERENCES "public"."events" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Create index "idx_tasks_embargo" to table: "tasks"
CREATE INDEX "idx_tasks_embargo" ON "public"."tasks" ("embargo_until");
-- Create index "idx_tasks_type" to table: "tasks"
CREATE INDEX "idx_tasks_type" ON "public"."tasks" ("task_type");
-- Set comment to column: "task_type" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."task_type" IS 'The type of task. E.g. `event.created`';
-- Set comment to column: "embargo_until" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."embargo_until" IS 'If set, the task will not be processed until after this time.';
-- Set comment to column: "retry_count" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."retry_count" IS 'Number of times this task has been retried.';
-- Set comment to column: "payload" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."payload" IS 'The task payload. This will be a JSON object with task-specific data.';
-- Set comment to column: "event_id" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."event_id" IS 'Optional reference to an event that triggered this task.';
