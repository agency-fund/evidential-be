-- Modify "experiments" table
ALTER TABLE "public"."experiments" DROP COLUMN "design_spec", ALTER COLUMN "design_spec_fields" SET NOT NULL;
