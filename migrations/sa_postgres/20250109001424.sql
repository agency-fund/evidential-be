-- Create "apikeys" table
CREATE TABLE "public"."apikeys" (
  "id" character varying NOT NULL,
  "key" character varying NOT NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "apikeys_key_key" UNIQUE ("key")
);
-- Create "cache" table
CREATE TABLE "public"."cache" (
  "key" character varying NOT NULL,
  "value" character varying NOT NULL,
  PRIMARY KEY ("key")
);
-- Create "apikey_datasources" table
CREATE TABLE "public"."apikey_datasources" (
  "apikey_id" character varying NOT NULL,
  "datasource_id" character varying NOT NULL,
  PRIMARY KEY ("apikey_id", "datasource_id"),
  CONSTRAINT "apikey_datasources_apikey_id_fkey" FOREIGN KEY ("apikey_id") REFERENCES "public"."apikeys" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
