-- Backfill new columns holding stats parameters for frequentist experiments using the design_spec values.
UPDATE experiments
SET power = (design_spec->'power')::float8,
    alpha = (design_spec->'alpha')::float8,
    fstat_thresh = (design_spec->'fstat_thresh')::float8;

