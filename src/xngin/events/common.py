from pydantic import BaseModel, ConfigDict


class BaseEventModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
