-- Snapshots were previously inserted as 'pending' and processed later by a separate transaction. The worker now
-- records each snapshot as 'success' or 'failed' in a single transaction, so any row still 'pending' was abandoned
-- before it could be processed and would otherwise be reported as running forever. This must run before the next
-- migration adds a CHECK constraint that rejects 'pending'.
UPDATE "public"."snapshots"
SET "status" = 'failed',
    "message" = 'Abandoned before the snapshot could be processed.'
WHERE "status" = 'pending';
