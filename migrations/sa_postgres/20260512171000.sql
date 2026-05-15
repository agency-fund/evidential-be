-- Backfill experiments.desired_n for preassigned experiments: set to total arm_assignment rows per experiment.
UPDATE experiments AS e
SET desired_n = (
    SELECT count(*)::integer
    FROM arm_assignments AS aa
    WHERE aa.experiment_id = e.id
)
WHERE e.experiment_type = 'freq_preassigned'
  AND e.desired_n IS NULL;
