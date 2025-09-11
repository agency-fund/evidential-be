-- Modify "users" table
ALTER TABLE "public"."users" ADD COLUMN "last_logout" timestamptz NOT NULL DEFAULT to_timestamp((0)::double precision);
