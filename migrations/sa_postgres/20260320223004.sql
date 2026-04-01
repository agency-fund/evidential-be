-- Create "arm_stats" table
CREATE TABLE "public"."arm_stats" (
  "arm_id" character varying(36) NOT NULL,
  "population" integer NOT NULL DEFAULT 0,
  PRIMARY KEY ("arm_id"),
  CONSTRAINT "arm_stats_arm_id_fkey" FOREIGN KEY ("arm_id") REFERENCES "public"."arms" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
