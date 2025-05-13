-- Migrate Experiment.audience_spec participant_type and filters json into the design_spec json
-- If 'design_spec' is NULL, it initializes it as an empty JSON object before merging.
UPDATE experiments
SET design_spec = COALESCE(design_spec, '{}'::jsonb) || jsonb_build_object(
    'participant_type', audience_spec ->> 'participant_type',
    'filters', audience_spec -> 'filters'
)
WHERE audience_spec IS NOT NULL;
