-- Backfill population counts from existing arm_assignments
INSERT INTO "public"."arm_stats" (arm_id, population)
SELECT arm_id, COUNT(*) FROM "public"."arm_assignments" GROUP BY arm_id;
-- Backfill population counts from existing draws
INSERT INTO "public"."arm_stats" (arm_id, population)
SELECT arm_id, COUNT(*) FROM "public"."draws" GROUP BY arm_id;

