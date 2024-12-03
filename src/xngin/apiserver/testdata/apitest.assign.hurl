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
  "assignments": [
    {
      "id": 710929,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 730934,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 440263,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 410077,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 486195,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 484970,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 393637,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 122169,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 155035,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 295539,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 552382,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 947098,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 431735,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 866648,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 407190,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 248783,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 76948,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 171643,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 302146,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 812175,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 154001,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 12955,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 35426,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 237616,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 210073,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 179916,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 65425,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 469679,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 510222,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 23560,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 773941,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 865084,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 33392,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 310074,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 198781,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 776353,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 996915,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 830767,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 956625,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 895588,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    }
  ],
  "balance_ok": false,
  "denominator_df": 33,
  "description": "Sample experiment description.",
  "experiment_id": "0897bbd7-dacd-41b4-b164-7a1950fa61f2",
  "f_statistic": 7.367106128752492,
  "numerator_df": 7,
  "p_value": 2.4555756260924433e-05,
  "sample_size": 40
}
```
