-- Drop foreign key constraint
-- linus: Need to drop this temporarily due to error "pq: foreign key constraint
-- "arm_assignments_experiment_id_fkey" cannot be implemented"
ALTER TABLE "public"."arm_assignments" DROP CONSTRAINT "arm_assignments_experiment_id_fkey";

-- Modify "arm_assignments" table
ALTER TABLE "public"."arm_assignments" ALTER COLUMN "experiment_id" TYPE character varying(36), ALTER COLUMN "arm_id" TYPE character varying(36);

-- Modify "experiments" table
ALTER TABLE "public"."experiments" ALTER COLUMN "id" TYPE character varying(36);

-- Recreate foreign key constraint
ALTER TABLE "public"."arm_assignments" ADD CONSTRAINT "arm_assignments_experiment_id_fkey"
    FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("id") ON DELETE CASCADE;
