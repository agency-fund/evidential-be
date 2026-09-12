-- Remove the "participants" key from the serialized DatasourceConfig in "datasources"."config".
-- RemoteDatabaseConfig no longer declares participant types, and ConfigBaseModel sets extra="forbid",
-- so any config that retains this key would fail validation in Datasource.get_config().
UPDATE "public"."datasources"
SET "config" = "config" - 'participants'
WHERE "config" ? 'participants';
