-- Seed each periodic job's last success from the work done before the worker existed, so the first deployment
-- doesn't snapshot or pull every experiment at once. The migration time stands in for the last attempt, so windows
-- that already closed get no final run.
INSERT INTO "public"."job_statuses" ("experiment_id", "job_name", "last_success_at", "last_run_at")
SELECT "experiment_id", 'snapshot', max("updated_at"), now()
FROM "public"."snapshots"
WHERE "status" = 'success'
GROUP BY "experiment_id";
INSERT INTO "public"."job_statuses" ("experiment_id", "job_name", "last_success_at", "last_run_at")
SELECT d."experiment_id", 'dwh_pull', max(d."observed_at"), now()
FROM "public"."draws" d
JOIN "public"."experiments" e ON e."id" = d."experiment_id"
WHERE e."experiment_type" = 'mab_online_dwh' AND d."observed_at" IS NOT NULL
GROUP BY d."experiment_id";
