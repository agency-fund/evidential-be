-- Rename a column from "draw_datetime_utc" to "created_at"
ALTER TABLE "public"."draws" RENAME COLUMN "draw_datetime_utc" TO "created_at";
-- Rename a column from "observed_datetime_utc" to "observed_at"
ALTER TABLE "public"."draws" RENAME COLUMN "observed_datetime_utc" TO "observed_at";
-- Modify "draws" table
ALTER TABLE "public"."draws" DROP CONSTRAINT "draws_pkey", ADD COLUMN "participant_id" character varying(255) NOT NULL, ADD COLUMN "participant_type" character varying(255) NOT NULL, ADD PRIMARY KEY ("participant_id"), DROP COLUMN "id", ALTER COLUMN "observed_at" DROP DEFAULT, ADD COLUMN "current_mu" double precision[] NULL, ADD COLUMN "current_covariance" double precision[] NULL, ADD COLUMN "current_alpha" double precision NULL, ADD COLUMN "current_beta" double precision NULL;
