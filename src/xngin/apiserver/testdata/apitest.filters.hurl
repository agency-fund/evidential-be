GET http://localhost:8000/filters?unit_type=test_unit_type
Config-ID: testing
HTTP 200

```json
[
  {
    "data_type": "double precision",
    "description": "",
    "distinct_values": [
      "9778.910653027706",
      "120621.36985115612"
    ],
    "filter_name": "baseline_income",
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "double precision",
    "description": "",
    "distinct_values": [
      "8466.284540241537",
      "96941.0"
    ],
    "filter_name": "current_income",
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "character varying",
    "description": "",
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
    "description": "",
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
    "description": "",
    "distinct_values": [
      "0",
      "1"
    ],
    "filter_name": "is_engaged",
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "",
    "distinct_values": [
      "0",
      "1"
    ],
    "filter_name": "is_onboarded",
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "",
    "distinct_values": [
      "1",
      "1"
    ],
    "filter_name": "is_recruited",
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "",
    "distinct_values": [
      "0",
      "1"
    ],
    "filter_name": "is_registered",
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "integer",
    "description": "",
    "distinct_values": [
      "0",
      "1"
    ],
    "filter_name": "is_retained",
    "relations": [
      "between"
    ]
  }
]

```
