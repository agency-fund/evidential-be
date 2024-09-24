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
    "data_type": "boolean",
    "description": "Engaged?",
    "distinct_values": [
      "0",
      "1"
    ],
    "filter_name": "is_engaged",
    "max": null,
    "min": null,
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "boolean",
    "description": "Onboarded?",
    "distinct_values": [
      "0",
      "1"
    ],
    "filter_name": "is_onboarded",
    "max": null,
    "min": null,
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "boolean",
    "description": "Recruited?",
    "distinct_values": [
      "1"
    ],
    "filter_name": "is_recruited",
    "max": null,
    "min": null,
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "boolean",
    "description": "Registered?",
    "distinct_values": [
      "0",
      "1"
    ],
    "filter_name": "is_registered",
    "max": null,
    "min": null,
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "boolean",
    "description": "Retained?",
    "distinct_values": [
      "0",
      "1"
    ],
    "filter_name": "is_retained",
    "max": null,
    "min": null,
    "relations": [
      "includes",
      "excludes"
    ]
  }
]
```
