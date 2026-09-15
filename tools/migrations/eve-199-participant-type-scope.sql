-- Scope of the participant-type removal (eve-199). READ ONLY: run against production before migrating.
--
-- Covers the two migrations on this branch:
--   20260901165333.sql  DROP TABLE participant_types_inspected
--   20260901184333.sql  UPDATE datasources SET config = config - 'participants'

-- 1. Summary: how many datasources are touched, and how much config is discarded.
SELECT
    count(*)                                                                 AS datasources_total,
    count(*) FILTER (WHERE config ? 'participants')                          AS rows_updated,
    count(*) FILTER (WHERE jsonb_array_length(config -> 'participants') > 0) AS with_any_ptype,
    coalesce(sum(jsonb_array_length(config -> 'participants')), 0)           AS ptype_defs_dropped
FROM public.datasources;

-- 2. Detail: every participant type definition that migration 20260901184333 discards.
--    `hidden = false` means it was user-authored and visible in the UI, so these are the
--    definitions a human may notice going away.
SELECT
    o.name                                        AS organization,
    ds.name                                       AS datasource,
    ds.id                                         AS datasource_id,
    ds.config -> 'dwh' ->> 'driver'               AS dwh_driver,
    p ->> 'participant_type'                      AS participant_type,
    p ->> 'table_name'                            AS table_name,
    coalesce((p ->> 'hidden')::boolean, false)    AS hidden,
    jsonb_array_length(p -> 'fields')             AS n_fields,
    (SELECT count(*) FROM public.experiments e
      WHERE e.datasource_id = ds.id
        AND e.datasource_table = p ->> 'table_name')  AS experiments_on_this_table
FROM public.datasources ds
JOIN public.organizations o ON o.id = ds.organization_id
CROSS JOIN LATERAL jsonb_array_elements(ds.config -> 'participants') AS p
ORDER BY hidden, organization, datasource, participant_type;

-- 3. Sanity check: any datasources.config keys other than the expected three? If this returns
--    rows, those configs will still fail validation after the migration (ConfigBaseModel is
--    extra="forbid") and need their own cleanup.
SELECT ds.id AS datasource_id, ds.name AS datasource, k AS unexpected_config_key
FROM public.datasources ds
CROSS JOIN LATERAL jsonb_object_keys(ds.config) AS k
WHERE k NOT IN ('type', 'dwh', 'participants')
ORDER BY datasource_id, unexpected_config_key;

-- 4. Cached inspection rows dropped with the participant_types_inspected table.
SELECT
    count(*)                     AS rows_dropped,
    count(DISTINCT datasource_id) AS datasources_affected,
    max(response_last_updated)   AS most_recent_inspection
FROM public.participant_types_inspected;
