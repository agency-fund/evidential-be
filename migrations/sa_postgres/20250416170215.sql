-- Create "webhooks" table
CREATE TABLE "public"."webhooks" (
  "id" character varying NOT NULL,
  "type" character varying NOT NULL,
  "url" character varying NOT NULL,
  "auth_token" character varying NULL,
  "organization_id" character varying NOT NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "webhooks_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION
);
-- Set comment to column: "type" on table: "webhooks"
COMMENT ON COLUMN "public"."webhooks"."type" IS 'The type of webhook; e.g. experiment.created';
-- Set comment to column: "url" on table: "webhooks"
COMMENT ON COLUMN "public"."webhooks"."url" IS 'The URL to post the event to.';
-- Set comment to column: "auth_token" on table: "webhooks"
COMMENT ON COLUMN "public"."webhooks"."auth_token" IS 'The authorization token.';
