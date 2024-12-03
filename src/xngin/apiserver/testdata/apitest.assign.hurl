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
      "id": 101638,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 765334,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 144894,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 364015,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 662027,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 24098,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 612541,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 709857,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 714173,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 223979,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 316979,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 349088,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 904464,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 725801,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 525958,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 698018,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 740855,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 523605,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 716737,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 995199,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 449677,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 758356,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 396379,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 531374,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 275539,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 591264,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 347822,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 727298,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 756198,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 638281,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 871555,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 587848,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 541539,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 748975,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 920612,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 985629,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 680188,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 283187,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "1"
        }
      ],
      "treatment_assignment": "1"
    },
    {
      "id": 477955,
      "strata": [
        {
          "strata_name": "is_onboarded",
          "strata_value": "0"
        }
      ],
      "treatment_assignment": "0"
    },
    {
      "id": 924026,
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
  "f_statistic": 5.9651128135123805,
  "numerator_df": 7,
  "p_value": 0.00015424592986257313,
  "sample_size": 40
}
```
