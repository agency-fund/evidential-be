UPDATE datasources SET config = config - 'webhook_config' WHERE jsonb_path_exists(config, '$.webhook_config');
