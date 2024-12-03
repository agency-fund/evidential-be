POST http://localhost:8000/commit?user_id=commituser
Config-ID: customer-test
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
    "participant_type": "groups",
    "filters": [
      {
        "filter_name": "is_onboarded",
        "relation": "includes",
        "value": [
          "TRUE"
        ]
      }
    ]
  },
  "experiment_assignment": {
    "f_statistic": 1.1153,
    "numerator_df": 523,
    "denominator_df": 476,
    "p_value": 0.1121,
    "balance_ok": false,
    "experiment_id": "df6456bf-09c5-4d7f-82f1-75679db54477",
    "description": "Sample experiment description.",
    "sample_size": 1000,
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
  "status_code": 200,
  "body": {
    "status": true,
    "message": "Success!",
    "data": {
      "s3_file_name": "https://testbucket.s3.ap-south-1.amazonaws.com/a_b_experiment/foo1.csv"
    }
  }
}
```
