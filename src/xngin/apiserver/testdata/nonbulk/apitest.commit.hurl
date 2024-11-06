POST http://localhost:8000/commit?user_id=commituser
Config-ID: testing
```json
{
  "design_spec": {
    "experiment_id": "0897bbd7-dacd-41b4-b164-7a1950fa61f2",
    "experiment_name": "Sample experiment name.",
    "description": "Sample experiment description.",
    "arms": [
      {
        "arm_name": "Control",
        "arm_id": "699a500b-41d6-4cba-b254-60b9493717cb"
      },
      {
        "arm_name": "Treatment",
        "arm_id": "5bc829fb-ffa6-456c-9e54-d26ba8d1031d"
      }
    ],
    "start_date": "2024-11-06",
    "end_date": "2024-11-20",
    "strata_cols": [
      "active_experiment_ids"
    ],
    "power": 0.8,
    "alpha": 0.05,
    "fstat_thresh": 0.5,
    "metrics": [
      {
        "metric_name": "chat_msgs_hour",
        "metric_pct_change": 0.1
      }
    ]
  },
  "audience_spec": {
    "type": "groups",
    "filters": [
      {
        "filter_name": "onboarded",
        "relation": "includes",
        "value": [
          "TRUE"
        ]
      }
    ]
  },
  "experiment_assignment": {
    "f_stat": "NaN",
    "numerator_df": 66,
    "denominator_df": 0,
    "p._value": "NaN",
    "balance_ok": {},
    "experiment_id": "test-eea257d2-4a3f-46d9-b3dd-7a1d3f6c8d47",
    "description": "Sample experiment description.",
    "sample_size": 100,
    "assignments": [
      {
        "groups_id": 35634,
        "treatment_assignment": "Treatment1",
        "strata": [
          {
            "strata_name": "participants",
            "strata_value": "4"
          },
          {
            "strata_name": "video_msgs_hour",
            "strata_value": "0"
          }
        ]
      }
    ]
  }
}
```
HTTP 200
```json
{
  "status": true,
  "message": "Success!",
  "data": {
    "s3_file_name": "https://testbucket.s3.ap-south-1.amazonaws.com/a_b_experiment/foo1.csv"
  }
}
```
