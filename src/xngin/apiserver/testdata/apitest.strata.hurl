GET http://localhost:8000/strata?unit_type=test_unit_type
Config-ID: testing
HTTP 200
```json
[
  {
    "column_name": "baseline_income",
    "data_type": "double precision",
    "description": "Baseline Income",
    "strata_group": ""
  },
  {
    "column_name": "current_income",
    "data_type": "double precision",
    "description": "Current Income",
    "strata_group": ""
  },
  {
    "column_name": "ethnicity",
    "data_type": "character varying",
    "description": "The ethnicity",
    "strata_group": "TODO"
  },
  {
    "column_name": "gender",
    "data_type": "character varying",
    "description": "Gender",
    "strata_group": "TODO"
  },
  {
    "column_name": "is_engaged",
    "data_type": "boolean",
    "description": "Engaged?",
    "strata_group": ""
  },
  {
    "column_name": "is_onboarded",
    "data_type": "boolean",
    "description": "Onboarded?",
    "strata_group": ""
  },
  {
    "column_name": "is_recruited",
    "data_type": "boolean",
    "description": "Recruited?",
    "strata_group": ""
  },
  {
    "column_name": "is_registered",
    "data_type": "boolean",
    "description": "Registered?",
    "strata_group": ""
  },
  {
    "column_name": "is_retained",
    "data_type": "boolean",
    "description": "Retained?",
    "strata_group": ""
  }
]
```
