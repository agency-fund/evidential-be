-- Set comment to column: "task_type" on table: "tasks"
COMMENT ON COLUMN "public"."tasks"."task_type" IS 'The type of task. E.g. `experiment.created`';
-- Set comment to column: "type" on table: "webhooks"
COMMENT ON COLUMN "public"."webhooks"."type" IS 'The type of webhook; e.g. experiment.created. These are user-visible arbitrary strings.';
-- Set comment to column: "url" on table: "webhooks"
COMMENT ON COLUMN "public"."webhooks"."url" IS 'The URL to post the event to. The payload body depends on the type of webhook.';
-- Set comment to column: "auth_token" on table: "webhooks"
COMMENT ON COLUMN "public"."webhooks"."auth_token" IS 'The token that will be sent in the Authorization header.';
