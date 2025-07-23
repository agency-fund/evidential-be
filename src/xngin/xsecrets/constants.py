# Specifies the active encryption backend.
ENV_XNGIN_SECRETS_BACKEND = "XNGIN_SECRETS_BACKEND"

# Google Cloud KMS.
# Key URI format: gcp-kms://projects/*/locations/*/keyRings/*/cryptoKeys/*
ENV_XNGIN_SECRETS_GCP_CREDENTIALS = "XNGIN_SECRETS_GCP_CREDENTIALS"
ENV_XNGIN_SECRETS_GCP_KMS_KEY_URI = "XNGIN_SECRETS_GCP_KMS_KEY_URI"

# Pynacl key for local encryption. Value generated with `xngin-cli create-nacl-keyset`.
ENV_XNGIN_SECRETS_NACL_KEYSET = "XNGIN_SECRETS_NACL_KEYSET"

# Serialized ciphertexts are prefixed with this value so that it is easily recognizable as
# an encrypted value.
SERIALIZED_ENCRYPTED_VALUE_PREFIX = "$encrypted$:"
