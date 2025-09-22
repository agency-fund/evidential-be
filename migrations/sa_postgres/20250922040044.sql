UPDATE "public"."apikeys" SET "created_at" = to_timestamp(0);
UPDATE "public"."datasources" SET "created_at" = to_timestamp(0);
UPDATE "public"."organizations" SET "created_at" = to_timestamp(0);
UPDATE "public"."user_organizations" SET "created_at" = to_timestamp(0);
UPDATE "public"."users" SET "created_at" = to_timestamp(0);
UPDATE "public"."webhooks" SET "created_at" = to_timestamp(0);
