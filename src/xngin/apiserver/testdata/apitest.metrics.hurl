GET http://localhost:8000/metrics?unit_type=test_unit_type
Config-ID: testing
HTTP 200
```json
[
  {
    "column_name": "current_income",
    "data_type": "double precision",
    "table_name": "test_unit_type"
  },
  {
    "column_name": "is_engaged",
    "data_type": "integer",
    "table_name": "test_unit_type"
  },
  {
    "column_name": "is_onboarded",
    "data_type": "integer",
    "table_name": "test_unit_type"
  },
  {
    "column_name": "is_recruited",
    "data_type": "integer",
    "table_name": "test_unit_type"
  },
  {
    "column_name": "is_registered",
    "data_type": "integer",
    "table_name": "test_unit_type"
  },
  {
    "column_name": "is_retained",
    "data_type": "integer",
    "table_name": "test_unit_type"
  }
]
```
