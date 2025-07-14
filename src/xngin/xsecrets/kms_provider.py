from tink import aead

from xngin.xsecrets.provider import Provider


class KmsProvider(Provider):
    """Implements a Provider for a KMS-backed Tink envelope encryption provider.

    Reference: https://cloud.google.com/kms/docs/encrypt-decrypt
    """

    def __init__(self, *, variant: str, remote_aead: aead.Aead):
        """Constructs a KmsProvider.

        :param variant: Informative and unique identifier for this provider (e.g. "gcpkms")
        :param remote_aead: The algorithm used to encrypt the encryption key (DEK).
        """
        self.variant = variant
        self.env_aead = aead.KmsEnvelopeAead(
            aead.aead_key_templates.AES128_GCM, remote_aead
        )

    def name(self) -> str:
        return self.variant

    def encrypt(self, pt: bytes, aad: bytes) -> bytes:
        return self.env_aead.encrypt(pt, aad)

    def decrypt(self, ct: bytes, aad: bytes) -> bytes:
        return self.env_aead.decrypt(ct, aad)
