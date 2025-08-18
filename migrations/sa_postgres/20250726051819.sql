-- Modify "experiments" table
ALTER TABLE "public"."experiments" ALTER COLUMN "design_spec_fields" DROP NOT NULL, ADD COLUMN "n_trials" integer NOT NULL DEFAULT 0, ADD COLUMN "prior_type" character varying NULL, ADD COLUMN "reward_type" character varying NULL;
-- Set comment to column: "experiment_type" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."experiment_type" IS NULL;
-- Create "context" table
CREATE TABLE "public"."context" (
  "id" character varying NOT NULL,
  "experiment_id" character varying(36) NOT NULL,
  "name" character varying(255) NOT NULL,
  "description" character varying(2000) NOT NULL,
  "value_type" character varying NOT NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "context_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Drop index "ix_arms_experiment_id" from table: "arms"
DROP INDEX "public"."ix_arms_experiment_id";
-- Modify "arms" table
ALTER TABLE "public"."arms" ADD COLUMN "mu_init" double precision NULL, ADD COLUMN "sigma_init" double precision NULL, ADD COLUMN "mu" double precision[] NULL, ADD COLUMN "covariance" double precision[] NULL, ADD COLUMN "alpha_init" double precision NULL, ADD COLUMN "beta_init" double precision NULL, ADD COLUMN "alpha" double precision NULL, ADD COLUMN "beta" double precision NULL;
-- Create "draws" table
CREATE TABLE "public"."draws" (
  "experiment_id" character varying(36) NOT NULL,
  "participant_id" character varying(255) NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "observed_at" timestamptz NULL,
  "observation_type" character varying NULL,
  "participant_type" character varying(255) NOT NULL,
  "arm_id" character varying(36) NOT NULL,
  "outcome" double precision NULL,
  "context_vals" double precision[] NULL,
  "current_mu" double precision[] NULL,
  "current_covariance" double precision[] NULL,
  "current_alpha" double precision NULL,
  "current_beta" double precision NULL,
  PRIMARY KEY ("experiment_id", "participant_id"),
  CONSTRAINT "draws_arm_id_fkey" FOREIGN KEY ("arm_id") REFERENCES "public"."arms" ("id") ON UPDATE NO ACTION ON DELETE CASCADE,
  CONSTRAINT "draws_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
