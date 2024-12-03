POST http://localhost:8000/assign?participant_type=test_participant_type&chosen_n=40&random_state=42
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
      "is_onboarded"
    ],
    "power": 0.8,
    "alpha": 0.05,
    "fstat_thresh": 0.5,
    "metrics": [
      {
        "metric_name": "current_income",
        "metric_pct_change": 0.1
      }
    ]
  },
  "audience_spec": {
    "participant_type": "test_participant_type",
    "filters": []
  }
}
```
HTTP 200
```json
{
  "f_statistic": 0.5692826346438268,
  "numerator_df": 13,
  "denominator_df": 26,
  "p_value": 0.8559923122792419,
  "balance_ok": true,
  "experiment_id": "0897bbd7-dacd-41b4-b164-7a1950fa61f2",
  "description": "Sample experiment description.",
  "sample_size": 40,
  "assignments": [
    {
      "id": "b767716b-f388-4cd9-a18a-08c4916ce26f",
      "treatment_assignment": "0",
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ]
    },
    {
      "id": "b767716b-f388-4cd9-a18a-08c4916ce26g",
      "treatment_assignment": "todo",
      "strata": [
        {
          "strata_name": "todo",
          "strata_value": "todo"
        }
      ]
    }
  ]
}
```
