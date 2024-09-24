GET http://localhost:8000/filters?unit_type=test_unit_type
Config-ID: testing
HTTP 200
```json
[
  {
    "data_type": "double precision",
    "description": "Baseline Income",
    "distinct_values": null,
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
    "distinct_values": null,
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
    "max": null,
    "min": null,
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
    "max": null,
    "min": null,
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "integer",
    "description": "Engaged?",
    "distinct_values": null,
    "filter_name": "is_engaged",
    "max": 1.0,
    "min": 0.0,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "Onboarded?",
    "distinct_values": null,
    "filter_name": "is_onboarded",
    "max": 1.0,
    "min": 0.0,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "Recruited?",
    "distinct_values": null,
    "filter_name": "is_recruited",
    "max": 1.0,
    "min": 1.0,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "Registered?",
    "distinct_values": null,
    "filter_name": "is_registered",
    "max": 1.0,
    "min": 0.0,
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "Retained?",
    "distinct_values": null,
    "filter_name": "is_retained",
    "max": 1.0,
    "min": 0.0,
    "relations": [
      "between"
    ]
  }
]
```
