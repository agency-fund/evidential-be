-- Create "snapshots" table
CREATE TABLE "public"."snapshots" (
  "experiment_id" character varying(36) NOT NULL,
  "id" character varying NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  "status" character varying(16) NOT NULL DEFAULT 'pending',
  "message" character varying NULL,
  "data" jsonb NULL,
  PRIMARY KEY ("experiment_id", "id"),
  CONSTRAINT "snapshots_id_key" UNIQUE ("id"),
  CONSTRAINT "snapshots_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
