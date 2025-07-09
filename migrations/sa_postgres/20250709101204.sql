-- Modify "experiments" table
ALTER TABLE "public"."experiments" DROP COLUMN "experiment_type", ADD COLUMN "assignment_type" character varying NOT NULL;
-- Set comment to column: "assignment_type" on table: "experiments"
COMMENT ON COLUMN "public"."experiments"."assignment_type" IS 'Should be one of the AssignmentType literals.';
