-- Modify "apikeys" table
ALTER TABLE "public"."apikeys" ADD COLUMN "created_at" timestamptz NOT NULL DEFAULT now(), ADD COLUMN "updated_at" timestamptz NOT NULL DEFAULT now();
-- Modify "datasources" table
ALTER TABLE "public"."datasources" ADD COLUMN "created_at" timestamptz NOT NULL DEFAULT now(), ADD COLUMN "updated_at" timestamptz NOT NULL DEFAULT now();
-- Modify "organizations" table
ALTER TABLE "public"."organizations" ADD COLUMN "created_at" timestamptz NOT NULL DEFAULT now(), ADD COLUMN "updated_at" timestamptz NOT NULL DEFAULT now();
-- Modify "user_organizations" table
ALTER TABLE "public"."user_organizations" ADD COLUMN "created_at" timestamptz NOT NULL DEFAULT now();
-- Modify "users" table
ALTER TABLE "public"."users" ADD COLUMN "created_at" timestamptz NOT NULL DEFAULT now(), ADD COLUMN "updated_at" timestamptz NOT NULL DEFAULT now();
-- Modify "webhooks" table
ALTER TABLE "public"."webhooks" ADD COLUMN "created_at" timestamptz NOT NULL DEFAULT now(), ADD COLUMN "updated_at" timestamptz NOT NULL DEFAULT now();
