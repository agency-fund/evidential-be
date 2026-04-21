-- Modify "turn_connections" table
ALTER TABLE "public"."turn_connections" ADD COLUMN "cached_journeys" jsonb NULL, ADD COLUMN "cached_journeys_updated_at" timestamptz NULL;
