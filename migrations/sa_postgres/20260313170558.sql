-- One-off cleanup as part of this migration away from participant types.
-- Should still consider deleting the abandoned experiment at time of abandonment or as a separate
-- periodic cleanup job.
delete from experiments where state = 'abandoned';
