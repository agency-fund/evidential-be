GET http://localhost:8000/strata?participant_type=test_participant_type
Config-ID: testing
HTTP 200
```json
[
  {
    "Extra Ad-Hoc Field": "",
    "column_name": "baseline_income",
    "data_type": "double precision",
    "description": "Baseline Income",
    "strata_group": "",
    "table": "test_participant_type"
  },
  {
    "Extra Ad-Hoc Field": "",
    "column_name": "current_income",
    "data_type": "double precision",
    "description": "Current Income",
    "strata_group": "",
    "table": "test_participant_type"
  },
  {
    "Extra Ad-Hoc Field": "adhoc2",
    "column_name": "ethnicity",
    "data_type": "character varying",
    "description": "The ethnicity",
    "strata_group": "TODO",
    "table": "test_participant_type"
  },
  {
    "Extra Ad-Hoc Field": "adhoc1",
    "column_name": "gender",
    "data_type": "character varying",
    "description": "Gender",
    "strata_group": "TODO",
    "table": "test_participant_type"
  },
  {
    "Extra Ad-Hoc Field": "",
    "column_name": "is_engaged",
    "data_type": "integer",
    "description": "Engaged?",
    "strata_group": "",
    "table": "test_participant_type"
  },
  {
    "Extra Ad-Hoc Field": "",
    "column_name": "is_onboarded",
    "data_type": "integer",
    "description": "Onboarded?",
    "strata_group": "",
    "table": "test_participant_type"
  },
  {
    "Extra Ad-Hoc Field": "",
    "column_name": "is_recruited",
    "data_type": "integer",
    "description": "Recruited?",
    "strata_group": "",
    "table": "test_participant_type"
  },
  {
    "Extra Ad-Hoc Field": "",
    "column_name": "is_registered",
    "data_type": "integer",
    "description": "Registered?",
    "strata_group": "",
    "table": "test_participant_type"
  },
  {
    "Extra Ad-Hoc Field": "",
    "column_name": "is_retained",
    "data_type": "integer",
    "description": "Retained?",
    "strata_group": "",
    "table": "test_participant_type"
  }
]
```
