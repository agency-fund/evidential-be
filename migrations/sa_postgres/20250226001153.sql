-- Modify "datasources" table
ALTER TABLE "public"."datasources" ADD COLUMN "table_list" jsonb NULL, ADD COLUMN "table_list_updated" timestamp NULL;
-- Set comment to column: "table_list" on table: "datasources"
COMMENT ON COLUMN "public"."datasources"."table_list" IS 'List of table names available in this datasource';
-- Set comment to column: "table_list_updated" on table: "datasources"
COMMENT ON COLUMN "public"."datasources"."table_list_updated" IS 'Timestamp of the last update to `inspected_tables`';
-- Create "datasource_tables_inspected" table
CREATE TABLE "public"."datasource_tables_inspected" (
  "datasource_id" character varying NOT NULL,
  "table_name" character varying NOT NULL,
  "response" jsonb NULL,
  "response_last_updated" timestamp NULL,
  PRIMARY KEY ("datasource_id", "table_name"),
  CONSTRAINT "datasource_tables_inspected_datasource_id_fkey" FOREIGN KEY ("datasource_id") REFERENCES "public"."datasources" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Set comment to column: "response" on table: "datasource_tables_inspected"
COMMENT ON COLUMN "public"."datasource_tables_inspected"."response" IS 'Serialized InspectDatasourceTablesResponse.';
-- Set comment to column: "response_last_updated" on table: "datasource_tables_inspected"
COMMENT ON COLUMN "public"."datasource_tables_inspected"."response_last_updated" IS 'Timestamp of the last update to `response`';
-- Create "participant_types_inspected" table
CREATE TABLE "public"."participant_types_inspected" (
  "datasource_id" character varying NOT NULL,
  "participant_type" character varying NOT NULL,
  "response" jsonb NULL,
  "response_last_updated" timestamp NULL,
  PRIMARY KEY ("datasource_id", "participant_type"),
  CONSTRAINT "participant_types_inspected_datasource_id_fkey" FOREIGN KEY ("datasource_id") REFERENCES "public"."datasources" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Set comment to column: "response" on table: "participant_types_inspected"
COMMENT ON COLUMN "public"."participant_types_inspected"."response" IS 'Serialized InspectParticipantTypesResponse.';
-- Set comment to column: "response_last_updated" on table: "participant_types_inspected"
COMMENT ON COLUMN "public"."participant_types_inspected"."response_last_updated" IS 'Timestamp of the last update to `response`';
