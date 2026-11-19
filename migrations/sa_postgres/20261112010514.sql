-- Create index "ix_users_email_lower" to table: "users"
CREATE UNIQUE INDEX "ix_users_email_lower" ON "public"."users" ((lower((email)::text)));
-- Create "domain_settings" table
CREATE TABLE "public"."domain_settings" (
  "id" character varying NOT NULL,
  "domain" character varying(128) NOT NULL,
  "allow_signup" boolean NOT NULL DEFAULT true,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "domain_settings_domain_key" UNIQUE ("domain"),
  CONSTRAINT "ck_domain_settings_domain_lowercase" CHECK ((domain)::text = lower((domain)::text))
);
-- Create "server_settings" table
CREATE TABLE "public"."server_settings" (
  "id" character varying(16) NOT NULL,
  "signup_mode" character varying(32) NOT NULL DEFAULT 'existing',
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "ck_server_settings_singleton" CHECK ((id)::text = 'singleton'::text)
);
