-- Create "events" table
CREATE TABLE "public"."events" (
  "id" character varying NOT NULL,
  "created_at" timestamptz NOT NULL,
  "type" character varying NOT NULL,
  "data" jsonb NOT NULL,
  "organization_id" character varying NOT NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "events_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION
);
-- Create index "event_stream" to table: "events"
CREATE INDEX "event_stream" ON "public"."events" ("organization_id", "created_at");
-- Set comment to column: "type" on table: "events"
COMMENT ON COLUMN "public"."events"."type" IS 'The type of event. E.g. `experiment.created`';
-- Set comment to column: "data" on table: "events"
COMMENT ON COLUMN "public"."events"."data" IS 'The event payload. This will always be a JSON object with a `type` field.';
