GET http://localhost:8000/metrics?unit_type=test_unit_type
Config-ID: testing
HTTP 200
```json
[
  {
    "column_name": "current_income",
    "data_type": "double precision",
    "description": "Current Income"
  },
  {
    "column_name": "is_engaged",
    "data_type": "integer",
    "description": "Engaged?"
  },
  {
    "column_name": "is_onboarded",
    "data_type": "integer",
    "description": "Onboarded?"
  },
  {
    "column_name": "is_recruited",
    "data_type": "integer",
    "description": "Recruited?"
  },
  {
    "column_name": "is_registered",
    "data_type": "integer",
    "description": "Registered?"
  },
  {
    "column_name": "is_retained",
    "data_type": "integer",
    "description": "Retained?"
  }
]
```
