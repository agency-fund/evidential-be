from pydantic import BaseModel, ConfigDict


class BaseEventModel(BaseModel):
    """Data stored in tables.Event.data should subclass this type.

    Subclasses must implement a type field and both methods.
    """

    model_config = ConfigDict(extra="forbid")

    def summarize(self) -> str:
        """Returns a human-readable one-sentence summary of this event."""
        return "Unrecognized event."

    def link(self) -> str | None:
        """Returns a navigable link to relevant details, or None.

        This is displayed in the UI so it may be an absolute path in the Dash UI.
        """
        return None
