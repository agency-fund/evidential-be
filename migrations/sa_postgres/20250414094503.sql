-- Drop primary key constraint first (which includes experiment_id)
ALTER TABLE "public"."arm_assignments" DROP CONSTRAINT "arm_assignments_pkey";

-- Drop foreign key constraint
ALTER TABLE "public"."arm_assignments" DROP CONSTRAINT "arm_assignments_experiment_id_fkey";

-- Modify "arm_assignments" table
ALTER TABLE "public"."arm_assignments" ALTER COLUMN "experiment_id" TYPE character varying(36), ALTER COLUMN "arm_id" TYPE character varying(36);

-- Modify "experiments" table
ALTER TABLE "public"."experiments" ALTER COLUMN "id" TYPE character varying(36);

-- Recreate primary key constraint
ALTER TABLE "public"."arm_assignments" ADD CONSTRAINT "arm_assignments_pkey"
    PRIMARY KEY ("experiment_id", "participant_id");

-- Recreate foreign key constraint
ALTER TABLE "public"."arm_assignments" ADD CONSTRAINT "arm_assignments_experiment_id_fkey"
    FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("id") ON DELETE CASCADE;
