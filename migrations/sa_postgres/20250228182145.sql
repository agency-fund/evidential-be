-- Modify "experiments" table
ALTER TABLE "public"."experiments" ADD CONSTRAINT "experiments_datasource_id_fkey" FOREIGN KEY ("datasource_id") REFERENCES "public"."datasources" ("id") ON UPDATE NO ACTION ON DELETE CASCADE;
