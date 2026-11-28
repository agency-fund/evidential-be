-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD COLUMN "update_frequency_hours" integer NULL;
-- Backfill: existing bandit experiments switch to scheduled arm updates every 4 hours.
UPDATE "public"."experiments" SET "update_frequency_hours" = 4
WHERE "experiment_type" IN ('mab_online', 'mab_online_dwh', 'cmab_online');
