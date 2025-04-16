-- Create "arms" table
CREATE TABLE "public"."arms" (
  "id" character varying(36) NOT NULL,
  "name" character varying(255) NOT NULL,
  "description" character varying(2000) NOT NULL,
  "experiment_id" character varying(36) NOT NULL,
  "organization_id" character varying NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "uix_arm_name_org" UNIQUE ("name", "organization_id"),
  CONSTRAINT "arms_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE,
  CONSTRAINT "arms_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
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
-- Finally, add the constraint on "arm_assignments" table
ALTER TABLE "public"."arm_assignments" ADD CONSTRAINT "arm_assignments_arm_id_fkey" FOREIGN KEY ("arm_id") REFERENCES "public"."arms" ("id") ON UPDATE NO ACTION ON DELETE CASCADE;
