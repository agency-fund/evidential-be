-- Modify "arm_assignments" table
ALTER TABLE "public"."arm_assignments" ADD COLUMN "cluster_key" character varying(255) NULL;
