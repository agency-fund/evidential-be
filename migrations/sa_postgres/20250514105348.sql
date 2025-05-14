-- Updates the 'design_spec' JSONB column in the 'experiments' table to pair with PR #400.
-- 1. Remove the 'strata_field_names' key, if it exists.
-- 2. Initialize the new 'strata' key to an empty list.

UPDATE experiments
SET design_spec = (design_spec - 'strata_field_names') || '{"strata": []}'::jsonb;
