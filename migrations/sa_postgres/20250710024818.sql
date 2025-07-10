-- Modify "experiments" table
ALTER TABLE "public"."experiments" ALTER COLUMN "design_spec_fields" DROP NOT NULL, ADD COLUMN "assignment_type" character varying NOT NULL, ADD COLUMN "n_trials" integer NOT NULL, ADD COLUMN "prior_type" character varying(50) NULL, ADD COLUMN "reward_type" character varying(50) NULL;
-- Set comment to column: "experiment_type" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."experiment_type" IS 'Should be one of the ExperimentType enums.';
-- Set comment to column: "assignment_type" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."assignment_type" IS 'Should be one of the AssignmentType literals.';
-- Create "context" table
CREATE TABLE "public"."context" (
  "id" character varying NOT NULL,
  "experiment_id" character varying(36) NOT NULL,
  "name" character varying(255) NOT NULL,
  "description" character varying(2000) NULL,
  "value_type" character varying NOT NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "context_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Drop index "ix_arms_experiment_id" from table: "arms"
DROP INDEX "public"."ix_arms_experiment_id";
-- Modify "arms" table
ALTER TABLE "public"."arms" ADD COLUMN "mu_init" double precision NULL, ADD COLUMN "sigma_init" double precision NULL, ADD COLUMN "mu" double precision[] NULL, ADD COLUMN "covariance" double precision[] NULL, ADD COLUMN "is_baseline" boolean NOT NULL, ADD COLUMN "alpha_init" double precision NULL, ADD COLUMN "beta_init" double precision NULL, ADD COLUMN "alpha" double precision NULL, ADD COLUMN "beta" double precision NULL;
-- Create "draws" table
CREATE TABLE "public"."draws" (
  "id" character varying NOT NULL,
  "draw_datetime_utc" timestamptz NOT NULL DEFAULT now(),
  "observed_datetime_utc" timestamptz NULL DEFAULT now(),
  "observation_type" character varying NULL,
  "experiment_id" character varying(36) NOT NULL,
  "arm_id" character varying(36) NOT NULL,
  "outcome" double precision NULL,
  "context_val" double precision[] NOT NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "draws_arm_id_fkey" FOREIGN KEY ("arm_id") REFERENCES "public"."arms" ("id") ON UPDATE NO ACTION ON DELETE CASCADE,
  CONSTRAINT "draws_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
