-- Keep experiment outcome tabs independent of shared raw datasource tabs.
ALTER TABLE "public"."experiments" ADD COLUMN "google_sheets_experiment_url" character varying NULL;
