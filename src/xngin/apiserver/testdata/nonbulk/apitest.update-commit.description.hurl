POST http://localhost:8000/update-commit?update_type=description
Config-ID: customer-test
```json
{
  "update_json": {
    "experiment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "description": "Sample experiment description and new description",
    "arms": [
      {
        "arm_name": "Control",
        "arm_id": "aa92c141-77d0-4dfe-ab03-af8559d7ba5a"
      },
      {
        "arm_name": "Treatment",
        "arm_id": "74a7e264-f0b9-4eb4-a2af-65deeeb9e105"
      }
    ]
  }
}
```
HTTP 200
```json
{
 "proxied_response": {
    "status": true,
    "message": "Experiment Commit Updated!"
  }
}
```
