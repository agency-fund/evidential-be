from typing import Annotated

from pydantic import Field
from xngin.events.experiment_created import ExperimentCreatedEvent
from xngin.events.webhook_sent import WebhookSentEvent

# EventDataTypes is the discriminated union type identifying valid tables.Event.data values.
type EventDataTypes = Annotated[
    WebhookSentEvent | ExperimentCreatedEvent, Field(discriminator="type")
]
