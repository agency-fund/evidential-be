-- Custom backfill for arms table
WITH src AS (
  SELECT
    d_arms.arm->>'arm_id' AS id,
    d_arms.arm->>'arm_name' AS name,
    d_arms.arm->>'arm_description' AS description,
    e.id AS experiment_id,
    d.organization_id,
    e.created_at,
    e.created_at AS updated_at
  FROM experiments e
  JOIN datasources d ON (e.datasource_id = d.id)
  CROSS JOIN LATERAL jsonb_array_elements(e.design_spec->'arms') AS d_arms(arm)
)
INSERT INTO arms (SELECT * FROM src ORDER BY created_at, id);
