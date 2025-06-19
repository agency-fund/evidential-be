# AWS KMS.
# Key URI format: aws-kms://arn:aws:kms:[region]:[account-id]:key/[key-id]
ENV_XNGIN_SECRETS_AWS_ACCESS_KEY_ID = "XNGIN_SECRETS_AWS_ACCESS_KEY_ID"
ENV_XNGIN_SECRETS_AWS_KEY_URI = "XNGIN_SECRETS_AWS_KEY_URI"
ENV_XNGIN_SECRETS_AWS_SECRET_ACCESS_KEY = "XNGIN_SECRETS_AWS_SECRET_ACCESS_KEY"

# Specifies the active encryption backend.
ENV_XNGIN_SECRETS_BACKEND = "XNGIN_SECRETS_BACKEND"

# Google Cloud KMS.
# Key URI format: gcp-kms://projects/*/locations/*/keyRings/*/cryptoKeys/*
ENV_XNGIN_SECRETS_GCP_CREDENTIALS = "XNGIN_SECRETS_GCP_CREDENTIALS"
ENV_XNGIN_SECRETS_GCP_KMS_KEY_URI = "XNGIN_SECRETS_GCP_KMS_KEY_URI"

# Tink keyset for local encryption. Value generated with `xngin-cli create-tink-keyset`.
ENV_XNGIN_SECRETS_TINK_KEYSET = "XNGIN_SECRETS_TINK_KEYSET"

# Serialized ciphertexts are prefixed with this value so that it is easily recognizable as
# an encrypted value similar to the ${secret:ENV} syntax we use in the static JSON settings files.
SERIALIZED_ENCRYPTED_VALUE_PREFIX = "$secret$:"
