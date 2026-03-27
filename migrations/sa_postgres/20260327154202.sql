CREATE INDEX ix_draws_arm_id_created_at
  ON draws (arm_id, created_at DESC)
  WHERE outcome IS NOT NULL;
