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
-- Modify "arm_assignments" table
ALTER TABLE "public"."arm_assignments" ADD CONSTRAINT "arm_assignments_arm_id_fkey" FOREIGN KEY ("arm_id") REFERENCES "public"."arms" ("id") ON UPDATE NO ACTION ON DELETE CASCADE;
