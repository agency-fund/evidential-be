-- Modify "snapshots" table
ALTER TABLE "public"."snapshots" ADD CONSTRAINT "ck_snapshots_status" CHECK ((status)::text = ANY ((ARRAY['success'::character varying, 'failed'::character varying])::text[])), ALTER COLUMN "status" DROP DEFAULT;
