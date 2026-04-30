-- Create "experiment_turn_configs" table
CREATE TABLE "public"."experiment_turn_configs" (
  "experiment_id" character varying(36) NOT NULL,
  "arm_journey_map" jsonb NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("experiment_id"),
  CONSTRAINT "experiment_turn_configs_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Create "turn_connections" table
CREATE TABLE "public"."turn_connections" (
  "organization_id" character varying NOT NULL,
  "encrypted_turn_api_token" character varying NOT NULL,
  "turn_api_token_preview" character varying(4) NOT NULL,
  "cached_journeys" jsonb NULL,
  "cached_journeys_updated_at" timestamptz NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("organization_id"),
  CONSTRAINT "turn_connections_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
