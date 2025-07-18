-- Rename a column from "draw_datetime_utc" to "created_at"
ALTER TABLE "public"."draws" RENAME COLUMN "draw_datetime_utc" TO "created_at";
-- Modify "draws" table
ALTER TABLE "public"."draws" DROP CONSTRAINT "draws_pkey", DROP COLUMN "id", DROP COLUMN "observed_datetime_utc", ADD COLUMN "participant_id" character varying(255) NOT NULL, ADD COLUMN "observed_at" timestamptz NULL, ADD COLUMN "participant_type" character varying(255) NOT NULL, ADD PRIMARY KEY ("participant_id");
