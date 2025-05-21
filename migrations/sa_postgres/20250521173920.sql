-- Backfill design_spec_fields from design_spec
UPDATE experiments
SET design_spec_fields = jsonb_build_object(
    'strata', design_spec->'strata',
    'metrics', design_spec->'metrics',
    'filters', design_spec->'filters'
)
WHERE design_spec_fields IS NULL;