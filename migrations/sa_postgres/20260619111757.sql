-- Rename a column from "cached_journeys" to "journeys_dict"
ALTER TABLE "public"."turn_connections" RENAME COLUMN "cached_journeys" TO "journeys_dict";
-- Modify "turn_connections" table
ALTER TABLE "public"."turn_connections" DROP COLUMN "cached_journeys_updated_at";
