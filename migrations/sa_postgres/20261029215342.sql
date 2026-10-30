-- Create "job_statuses" table
CREATE TABLE "public"."job_statuses" (
  "experiment_id" character varying(36) NOT NULL,
  "job_name" character varying(32) NOT NULL,
  "last_run_at" timestamptz NULL,
  "last_success_at" timestamptz NULL,
  "consecutive_failures" integer NOT NULL DEFAULT 0,
  "last_error" character varying(2048) NULL,
  "retry_not_before" timestamptz NULL,
  PRIMARY KEY ("experiment_id", "job_name"),
  CONSTRAINT "job_statuses_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
