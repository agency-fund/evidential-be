-- Modify "webhooks" table
ALTER TABLE "public"."webhooks" ALTER COLUMN "url" DROP NOT NULL, ADD COLUMN "direction" character varying NOT NULL DEFAULT 'outbound';
