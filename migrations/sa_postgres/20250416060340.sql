-- Modify "tasks" table
ALTER TABLE "public"."tasks" ALTER COLUMN "embargo_until" SET NOT NULL, ALTER COLUMN "embargo_until" SET DEFAULT now();
-- Set comment to column: "embargo_until" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."embargo_until" IS 'Time until which the task should not be processed. Defaults to created_at.';
