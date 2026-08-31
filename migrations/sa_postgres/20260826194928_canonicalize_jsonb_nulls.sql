-- Normalize JSON null to SQL NULL for the nullable JSONB columns that now use none_as_null=True.
UPDATE "public"."experiments"
SET "power_analyses" = CASE WHEN "power_analyses" = 'null'::jsonb THEN NULL ELSE "power_analyses" END,
    "balance_check" = CASE WHEN "balance_check" = 'null'::jsonb THEN NULL ELSE "balance_check" END
WHERE "power_analyses" = 'null'::jsonb OR "balance_check" = 'null'::jsonb;

UPDATE "public"."datasources"
SET "table_list" = NULL
WHERE "table_list" = 'null'::jsonb;

UPDATE "public"."turn_connections"
SET "journeys_dict" = NULL
WHERE "journeys_dict" = 'null'::jsonb;

UPDATE "public"."datasource_tables_inspected"
SET "response" = NULL
WHERE "response" = 'null'::jsonb;

UPDATE "public"."participant_types_inspected"
SET "response" = NULL
WHERE "response" = 'null'::jsonb;

UPDATE "public"."snapshots"
SET "data" = NULL
WHERE "data" = 'null'::jsonb;

UPDATE "public"."tasks"
SET "payload" = NULL
WHERE "payload" = 'null'::jsonb;
