-- Modify "arm_assignments" table
ALTER TABLE "public"."arm_assignments" ADD COLUMN "created_at" timestamptz NOT NULL DEFAULT now();
