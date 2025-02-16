-- Create "organizations" table
CREATE TABLE "public"."organizations" (
  "id" character varying NOT NULL,
  "name" character varying(255) NOT NULL,
  PRIMARY KEY ("id")
);
-- Create "datasources" table
CREATE TABLE "public"."datasources" (
  "id" character varying NOT NULL,
  "name" character varying(255) NOT NULL,
  "organization_id" character varying NOT NULL,
  "config" json NOT NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "datasources_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Set comment to column: "config" on table: "datasources"
COMMENT ON COLUMN "public"."datasources"."config" IS 'JSON serialized form of DatasourceConfig';
-- Modify "apikeys" table
ALTER TABLE "public"."apikeys" ADD COLUMN "datasource_id" character varying NOT NULL, ADD CONSTRAINT "apikeys_datasource_id_fkey" FOREIGN KEY ("datasource_id") REFERENCES "public"."datasources" ("id") ON UPDATE NO ACTION ON DELETE CASCADE;
-- Create "users" table
CREATE TABLE "public"."users" (
  "id" character varying NOT NULL,
  "email" character varying(255) NOT NULL,
  "iss" character varying(255) NULL,
  "sub" character varying(255) NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "users_email_key" UNIQUE ("email")
);
-- Create "user_organizations" table
CREATE TABLE "public"."user_organizations" (
  "user_id" character varying NOT NULL,
  "organization_id" character varying NOT NULL,
  PRIMARY KEY ("user_id", "organization_id"),
  CONSTRAINT "user_organizations_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations" ("id") ON UPDATE NO ACTION ON DELETE CASCADE,
  CONSTRAINT "user_organizations_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."users" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Drop "apikey_datasources" table
DROP TABLE "public"."apikey_datasources";
