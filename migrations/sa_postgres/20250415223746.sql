-- Modify "users" table
ALTER TABLE "public"."users" ADD COLUMN "is_privileged" boolean NOT NULL DEFAULT false;
-- Set comment to column: "is_privileged" on table: "users"
COMMENT ON COLUMN "public"."users"."is_privileged" IS 'True when this user is considered to be privileged.';
