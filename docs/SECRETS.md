# ☠️ Secrets and Where To Find Them<a name="%E2%98%A0%EF%B8%8F-secrets-and-where-to-find-them"></a>

<!-- mdformat-toc start --slug=github --maxlevel=3 --minlevel=1 -->

- [☠️ Secrets and Where To Find Them](#%E2%98%A0%EF%B8%8F-secrets-and-where-to-find-them)
  - [How are the encrypted fields encoded into stored Pydantic types?](#how-are-the-encrypted-fields-encoded-into-stored-pydantic-types)
  - [Configuration](#configuration)
    - [Selecting an Encryption Provider](#selecting-an-encryption-provider)
    - [Enabling "local" encryption](#enabling-local-encryption)
    - [Enabling AWS KMS](#enabling-aws-kms)
    - [Enabling Google Cloud KMS](#enabling-google-cloud-kms)

<!-- mdformat-toc end -->

Evidential encrypts customer-provided database credentials. Encrypting the credentials reduces the risk that
insecurities in the API server, database backups, or other operational practices will leak the credentials.

We have implemented encryption-at-rest for these values:

- BigQuery service account credentials
- Postgres/Redshift connection passwords

To support diverse deployment configurations, we support the following configurations:

| **Backend**                 | **Use Case**                                         | **Description**                                                                                                                                                               |
| --------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No-op (`noop`)              | Local development only.                              | No encryption is used.                                                                                                                                                        |
| Local (`local`)             | Traditional web hosts, self-hosting, or development. | Credentials are encrypted using a symmetric encryption key. The encryption key is readable by the API server.                                                                 |
| AWS KMS (`awskms`)          | When deploying on AWS.                               | Credentials are encrypted using AWS KMS envelope encryption. The encryption key is only usable by our server when the runtime infrastructure has access to the cloud service. |
| Google Cloud KMS (`gcpkms`) | When deploying on GCP.                               | Credentials are encrypted using GCP KMS envelope encryption. The encryption key is only usable by our server when the runtime infrastructure has access to the cloud service. |

Our encryption is implemented using industry standard AES128-GCM as implemented by
Google's [Tink](https://developers.google.com/tink) library with additional authenticated data (AAD) binding to
appropriate server-generated database identifiers. When using the KMS backends, we use
Tink's [Envelope Encryption](https://cloud.google.com/kms/docs/envelope-encryption)
implementation. Customer data is never sent to the cloud providers.

## How are the encrypted fields encoded into stored Pydantic types?<a name="how-are-the-encrypted-fields-encoded-into-stored-pydantic-types"></a>

Datasource configuration is modified with the `.set_config(new_config)` and
`.get_config()` methods on tables.Datasource. These methods invoke the encryption and decryption methods on the Pydantic
types that require them. This implementation allows most uses of the Pydantic types to be ignorant of the encryption,
isolating the encryption considerations to the storage tables.

To add new encrypted fields, consider adding them as top-level database fields and using a SQLAlchemy TypeDecorator
instead of serializing them as Pydantic/JSON fields. If you must, please follow the example set by the datasource
credentials.

## Configuration<a name="configuration"></a>

### Selecting an Encryption Provider<a name="selecting-an-encryption-provider"></a>

The various providers will initialize themselves automatically based on the presence
or absence of environment variables specific to that backend (see below). This allows the code to **decrypt** secrets
written by any of the properly configured backends, such that we could decrypt AWS KMS and GCP KMS and locally encrypted
values in a
single instance.

The encryption algorithm for new secrets is determined by the `XNGIN_SECRETS_BACKEND` environment variable.

The ability to decrypt from multiple providers and encrypt with a primary one is how we can support migration: to
migrate to a new encryption provider, change `XNGIN_SECRETS_BACKEND` to the
new provider, and then re-read and re-write all the encrypted values.

Choose **one** of these settings for your deployment:

```
XNGIN_SECRETS_BACKEND=local
XNGIN_SECRETS_BACKEND=awskms
XNGIN_SECRETS_BACKEND=gcpkms
```

### Enabling "local" encryption<a name="enabling-local-encryption"></a>

1. Set the `XNGIN_SECRETS_TINK_KEYSET` environment variable to the value returned by:

```shell
uv run xngin-cli create-tink-key
```

2. Set the `XNGIN_SECRETS_BACKEND` environment variable to `local`.

### Enabling AWS KMS<a name="enabling-aws-kms"></a>

> Note: The AWS KMS provider was written by Google, not Amazon, and it has some configuration quirks that are not
> idiomatic for AWS deployments. When running Evidential API server on AWS, consider improving our wrapper around
> Google's AWS KMS provider to acquire credentials using the temporary instance profile credentials provided via the
> metadata service rather than explicit access key configuration.

These variables will enable AWS KMS integration:

```
# AWS credentials with sufficient access to perform Encrypt and Decrypt on the key.
XNGIN_SECRETS_AWS_ACCESS_KEY_ID=...
XNGIN_SECRETS_AWS_SECRET_ACCESS_KEY=...

# The full AWS ARN of the key, prefixed with aws-kms://
XNGIN_SECRETS_AWS_KEY_URI=aws-kms://arn:aws:kms:[region]:[account-id]:key/[key-id]
```

To make AWS KMS the default encryption provider, set the `XNGIN_SECRETS_BACKEND` environment variable to `awskms`.

### Enabling Google Cloud KMS<a name="enabling-google-cloud-kms"></a>

Configuring the Google Cloud KMS service for use with Evidential involves multiple steps:

1. Ensuring your Google Cloud principal has sufficient privileges to perform KMS administration operations.
1. Enabling Google Cloud KMS on your account (if not already enabled)
1. Creating a service account for Evidential.
1. Creating a Google Cloud KMS keyring and key.
1. Granting access to the new service account to encrypt and decrypt with that key.
1. Configuring Evidential to use the new credentials and key.

The details of each step will vary based on the particulars of your deployment. For your convenience,
the steps to perform the above operations on an unconfigured Google Cloud account are below. If you already have
some Google Cloud configurations, you should adapt this script to your needs before running it. You will need a highly
privileged user to perform these operations.

```bash
PROJECT_ID="your-gcp-project"
REGION="us-west1"  # Choose the region closest to your servers
SERVICE_ACCOUNT_NAME="xngin-kms-sa"
SERVICE_ACCOUNT_DISPLAY_NAME="Xngin KMS Service Account"
KEYRING_NAME="xngin-keyring"
KEY_NAME="xngin-key"

# 1. Ensure your principal has the Cloud KMS Admin role
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:$(gcloud config get-value account)" \
  --role="roles/cloudkms.admin"

# 2. Create a new service account
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
  --display-name="$SERVICE_ACCOUNT_DISPLAY_NAME" \
  --project=$PROJECT_ID

# 3. Enable the Google Cloud KMS API
gcloud services enable cloudkms.googleapis.com --project=$PROJECT_ID

# 4. Create a keyring in the region closest to your servers
gcloud kms keyrings create $KEYRING_NAME \
  --location=$REGION \
  --project=$PROJECT_ID

# 5. Create a key
gcloud kms keys create $KEY_NAME \
  --location=$REGION \
  --keyring=$KEYRING_NAME \
  --purpose="encryption" \
  --protection-level="software" \
  --project=$PROJECT_ID

# 6. Grant the service account access to the key
gcloud kms keys add-iam-policy-binding $KEY_NAME \
  --location=$REGION \
  --keyring=$KEYRING_NAME \
  --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" \
  --project=$PROJECT_ID

# 7. Get the resource name for the key
KEY_URI="gcp-kms://projects/$PROJECT_ID/locations/$REGION/keyRings/$KEYRING_NAME/cryptoKeys/$KEY_NAME"
echo "Your KMS key URI is: $KEY_URI"

# 8. Create a service account key and save it to a file
gcloud iam service-accounts keys create service-account-key.json \
  --iam-account="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
  --project=$PROJECT_ID

# 9. Base64 encode the service account key
SA_CREDENTIALS=$(cat service-account-key.json | base64 -w 0)
echo "Your base64 encoded service account credentials are ready"
```

1. Configure the environment variables as follows:

```bash
# You can use the variables from the previous steps
export XNGIN_SECRETS_BACKEND=gcpkms
export XNGIN_SECRETS_GCP_CREDENTIALS=$SA_CREDENTIALS
export XNGIN_SECRETS_GCP_KMS_KEY_URI=$KEY_URI
```

1. If you are keeping these credentials locally, you can test these values with the `xngin-cli encrypt` and
   `xngin-cli decrypt` tools. These commands read the same
   environment variables that the API server does. Here's an example that demonstrates how to encrypt and decrypt "
   secretvalue":

```shell
# Encrypt the string "secretvalue" and then immediately decrypt it.
$ echo secretvalue | \
  uv run --env-file .env xngin-cli encrypt | \
  tee /tmp/ciphertext | \
  uv run --env-file .env xngin-cli decrypt
secretvalue
# Observe the serialized output format
$ cat /tmp/ciphertext
${secret:x:gcpkms:AAAAcwokAFUxnWVXyvell...
```

#### Debugging GCP KMS

gRPC logging verbosity can be adjusted with environment variables. For example, this will increase
the logging output:

```
# GRPC_VERBOSITY must be set to "debug" to enable tracing.
GRPC_VERBOSITY=debug
# One of these three settings is probably useful.
GRPC_TRACE=channel,subchannel,call_stream
GRPC_TRACE=tcp,http,secure_endpoint,transport_security
GRPC_TRACE=call_error,connectivity_state,pick_first,round_robin,glb
```

More details here: https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
