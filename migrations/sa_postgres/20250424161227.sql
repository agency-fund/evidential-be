-- Manual update of existing preassigned Experiment records by unpacking some values in our JSON.
UPDATE experiments
SET experiment_type = 'preassigned',
    participant_type = audience_spec->>'participant_type',
    name = design_spec->>'experiment_name',
    description = design_spec->>'description',
    design_spec = design_spec || '{"experiment_type": "preassigned"}';

-- Modify "experiments" table constraints now that we've backfilled
ALTER TABLE "public"."experiments" ALTER COLUMN "experiment_type" SET NOT NULL, ALTER COLUMN "participant_type" SET NOT NULL, ALTER COLUMN "name" SET NOT NULL, ALTER COLUMN "description" SET NOT NULL;
