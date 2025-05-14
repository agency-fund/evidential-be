-- Back-fill the events.data->>datasource_id field with the datasource of the corresponding experiment.
-- This corresponds to the change in ExperimentCreatedEvent in 5fda6fa.
UPDATE events
SET data = jsonb_set(events.data, '{datasource_id}', to_jsonb(experiments.datasource_id))
FROM experiments
WHERE events.data ->> 'experiment_id' = experiments.id
  AND events.type = 'experiment.created';
