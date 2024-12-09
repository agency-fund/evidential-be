GET http://localhost:8000/filters?participant_type=test_participant_type
Config-ID: testing
HTTP 200
```json
[
  {
    "data_type": "double precision",
    "description": "Baseline Income",
    "filter_name": "baseline_income",
    "max": 120621.36985115612,
    "min": 9778.910653027706,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "double precision",
    "description": "Current Income",
    "filter_name": "current_income",
    "max": 96941.0,
    "min": 8466.284540241537,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "character varying",
    "description": "The ethnicity",
    "distinct_values": [
      "Arabic",
      "Asian",
      "Black",
      "Hispanic",
      "Indian",
      "White"
    ],
    "filter_name": "ethnicity",
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "character varying",
    "description": "Gender",
    "distinct_values": [
      "Female",
      "Male"
    ],
    "filter_name": "gender",
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "integer",
    "description": "Engaged?",
    "filter_name": "is_engaged",
    "max": 1,
    "min": 0,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "Onboarded?",
    "filter_name": "is_onboarded",
    "max": 1,
    "min": 0,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "Recruited?",
    "filter_name": "is_recruited",
    "max": 1,
    "min": 1,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "Registered?",
    "filter_name": "is_registered",
    "max": 1,
    "min": 0,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "Retained?",
    "filter_name": "is_retained",
    "max": 1,
    "min": 0,
    "relations": [
      "between"
    ]
  }
]
```
