-- Modify "draws" table
ALTER TABLE "public"."draws" ADD COLUMN "autofailed_outcome" boolean NULL;
-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "enable_autofail" boolean NOT NULL DEFAULT false, ADD COLUMN "autofail_window" integer NOT NULL DEFAULT 24, ADD COLUMN "autofail_outcome_value" double precision NOT NULL DEFAULT 0;
-- Create "autofail_updates" table
CREATE TABLE "public"."autofail_updates" (
  "experiment_id" character varying(36) NOT NULL,
  "participant_id" character varying(255) NOT NULL,
  "id" character varying NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  "status" character varying(16) NOT NULL DEFAULT 'pending',
  "message" character varying NULL,
  "data" jsonb NULL,
  PRIMARY KEY ("experiment_id", "participant_id", "id"),
  CONSTRAINT "autofail_updates_id_key" UNIQUE ("id"),
  CONSTRAINT "autofail_updates_experiment_id_participant_id_fkey" FOREIGN KEY ("experiment_id", "participant_id") REFERENCES "public"."draws" ("experiment_id", "participant_id") ON UPDATE NO ACTION ON DELETE CASCADE
);
