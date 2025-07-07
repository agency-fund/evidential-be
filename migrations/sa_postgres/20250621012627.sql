-- Modify "webhooks" table
ALTER TABLE "public"."webhooks" ADD COLUMN "name" character varying NOT NULL DEFAULT '';
-- Create "experiment_webhooks" table
CREATE TABLE "public"."experiment_webhooks" (
  "experiment_id" character varying(36) NOT NULL,
  "webhook_id" character varying NOT NULL,
  PRIMARY KEY ("experiment_id", "webhook_id"),
  CONSTRAINT "experiment_webhooks_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE,
  CONSTRAINT "experiment_webhooks_webhook_id_fkey" FOREIGN KEY ("webhook_id") REFERENCES "public"."webhooks" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
