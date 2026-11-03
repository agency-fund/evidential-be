-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD CONSTRAINT "ck_experiments_end_not_before_start" CHECK (end_date >= start_date);
