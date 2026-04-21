-- Rename a column from "turn_api_token" to "encrypted_turn_api_token"
ALTER TABLE "public"."turn_connections" RENAME COLUMN "turn_api_token" TO "encrypted_turn_api_token";
-- Modify "turn_connections" table
ALTER TABLE "public"."turn_connections" ADD COLUMN "turn_api_token_preview" character varying(4) NOT NULL;
