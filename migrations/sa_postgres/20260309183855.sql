-- Modify "experiment_filters" table
ALTER TABLE "public"."experiment_filters" ADD CONSTRAINT "experiment_filters_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments" ("id") ON UPDATE NO ACTION ON DELETE CASCADE;
