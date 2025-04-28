-- Update experiments table to set balance_check column from assign_summary
UPDATE "public"."experiments"
SET balance_check = assign_summary->'balance_check'
WHERE assign_summary IS NOT NULL AND assign_summary->>'balance_check' IS NOT NULL;
