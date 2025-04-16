# Maximum length of fields that describe a value in human readable, free-form text.
MAX_LENGTH_OF_DESCRIPTION_VALUE = 2000

# Maximum length of an email address.
MAX_LENGTH_OF_EMAIL_VALUE = 64

# Maximum length of fields that contain identifiers such as primary keys, UUIDs, etc.
MAX_LENGTH_OF_ID_VALUE = 64

# Maximum length of fields that contain identifiers such as primary keys, UUIDs, etc.
MAX_LENGTH_OF_WEBHOOK_URL_VALUE = 500

# Maximum length of fields that contain names of things (such as experiment name, participant types, Arm names).
MAX_LENGTH_OF_NAME_VALUE = 100

# Maximum length of fields that contain unique identifiers for participants.
MAX_LENGTH_OF_PARTICIPANT_ID_VALUE = 64

# Maximum number of arms allowed in an experiment.
MAX_NUMBER_OF_ARMS = 10

# Maximum number of fields that may appear in any single list. Some types contain more than one list of fields, so
# there may be some multiple of this value in an object.
MAX_NUMBER_OF_FIELDS = 100

# Maximum number of filters we allow to define an audience.
MAX_NUMBER_OF_FILTERS = 20
