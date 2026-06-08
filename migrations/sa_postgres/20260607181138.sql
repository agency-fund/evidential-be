-- Modify "turn_connections" table
ALTER TABLE "public"."turn_connections" DROP COLUMN "cached_journeys", DROP COLUMN "cached_journeys_updated_at", ADD COLUMN "journeys_dict" jsonb NULL;
