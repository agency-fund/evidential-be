-- Modify "events" table
ALTER TABLE "public"."events" DROP CONSTRAINT "events_organization_id_fkey", ADD CONSTRAINT "events_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations" ("id") ON UPDATE NO ACTION ON DELETE CASCADE;
-- Modify "webhooks" table
ALTER TABLE "public"."webhooks" DROP CONSTRAINT "webhooks_organization_id_fkey", ADD CONSTRAINT "webhooks_organization_id_fkey" FOREIGN KEY ("organization_id") REFERENCES "public"."organizations" ("id") ON UPDATE NO ACTION ON DELETE CASCADE;
