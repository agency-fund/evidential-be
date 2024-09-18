GET http://localhost:8000/filters?group=sample_group
Config-ID: testing
HTTP 200
```json
[
  {
    "data_type": "character varying",
    "description": "",
    "distinct_values": [],
    "filter_name": "ethnicity",
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "character varying",
    "description": "",
    "distinct_values": [],
    "filter_name": "first_name",
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "character varying",
    "description": "",
    "distinct_values": [],
    "filter_name": "gender",
    "relations": [
      "includes",
      "excludes"
    ]
  },
  {
    "data_type": "double precision",
    "description": "",
    "distinct_values": [],
    "filter_name": "income",
    "relations": [
      "between"
    ]
  },
  {
    "data_type": "character varying",
    "description": "",
    "distinct_values": [],
    "filter_name": "last_name",
    "relations": [
      "includes",
      "excludes"
    ]
  }
]

```
