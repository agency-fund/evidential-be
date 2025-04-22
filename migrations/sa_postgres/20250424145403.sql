-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "experiment_type" character varying, ADD COLUMN "participant_type" character varying(255), ADD COLUMN "name" character varying(255), ADD COLUMN "description" character varying(2000);
-- Set comment to column: "experiment_type" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."experiment_type" IS 'Should be one of the ExperimentType literals.';
