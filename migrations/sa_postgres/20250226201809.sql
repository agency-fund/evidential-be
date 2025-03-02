-- Create enum type "experimentstate"
CREATE TYPE "public"."experimentstate" AS ENUM ('DESIGNING', 'ASSIGNED', 'ABANDONED', 'COMMITTED', 'ABORTED');
-- Modify "datasource_tables_inspected" table
ALTER TABLE "public"."datasource_tables_inspected" ALTER COLUMN "response_last_updated" TYPE timestamptz;
-- Modify "datasources" table
ALTER TABLE "public"."datasources" ALTER COLUMN "table_list_updated" TYPE timestamptz;
-- Modify "participant_types_inspected" table
ALTER TABLE "public"."participant_types_inspected" ALTER COLUMN "response_last_updated" TYPE timestamptz;
-- Create "experiments" table
CREATE TABLE "public"."experiments" (
  "id" uuid NOT NULL,
  "datasource_id" character varying(255) NOT NULL,
  "state" "public"."experimentstate" NOT NULL,
  "start_date" timestamptz NOT NULL,
  "end_date" timestamptz NOT NULL,
  "design_spec" jsonb NOT NULL,
  "audience_spec" jsonb NOT NULL,
  "power_analyses" jsonb NULL,
  "assign_summary" jsonb NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id")
);
-- Set comment to column: "start_date" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."start_date" IS 'Target start date of the experiment. Denormalized from design_spec.';
-- Set comment to column: "end_date" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."end_date" IS 'Target end date of the experiment. Denormalized from design_spec.';
-- Set comment to column: "design_spec" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."design_spec" IS 'JSON serialized form of DesignSpec.';
-- Set comment to column: "audience_spec" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."audience_spec" IS 'JSON serialized form of AudienceSpec.';
-- Set comment to column: "power_analyses" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."power_analyses" IS 'JSON serialized form of a PowerResponse. Not required since some experiments may not have data to run power analyses.';
-- Set comment to column: "assign_summary" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."assign_summary" IS 'JSON serialized form of AssignSummary.';
-- Create "arm_assignments" table
CREATE TABLE "public"."arm_assignments" (
  "experiment_id" uuid NOT NULL,
  "participant_id" character varying(255) NOT NULL,
  "participant_type" character varying(255) NOT NULL,
  "arm_id" uuid NOT NULL,
  "strata" jsonb NOT NULL,
  PRIMARY KEY ("experiment_id", "participant_id"),
  CONSTRAINT "arm_assignments_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Set comment to column: "strata" on table: "arm_assignments"
COMMENT ON COLUMN "public"."arm_assignments"."strata" IS 'JSON serialized form of a list of Strata objects (from Assignment.strata).';
