import hashlib
from pathlib import Path

PATH_TO_AMAZON_TRUST_CA_BUNDLE = (
    (Path(__file__).resolve().parent / "amazon-trust-ca-bundle.crt")
    .resolve(strict=True)
    .as_posix()
)

CERT_FILES_HASHES = {
    PATH_TO_AMAZON_TRUST_CA_BUNDLE: "36dba8e4b8041cd14b9d60158893963301bcbb92e1c456847784de2acb5bd550",
}


class CertsError(Exception):
    """Raised whenever there is an error relating to our stored CA certificates."""


def get_amazon_trust_ca_bundle_path():
    """Returns the absolute path on disk to the certificates issued by Amazon's CA."""
    assert_file_hash(
        PATH_TO_AMAZON_TRUST_CA_BUNDLE,
        CERT_FILES_HASHES[PATH_TO_AMAZON_TRUST_CA_BUNDLE],
    )
    return PATH_TO_AMAZON_TRUST_CA_BUNDLE


def assert_file_hash(file_path, expected_hash):
    """
    Verifies that a file exists and has the expected SHA256 hash.

    Args:
        file_path (str): Path to the file to verify
        expected_hash (str): Expected SHA256 hash of the file

    Raises:
        CertsError: If the file doesn't exist or the SHA256 hash doesn't match
    """
    path = Path(file_path)
    try:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError as err:
        raise CertsError(f"{file_path} not found") from err

    if actual_hash != expected_hash:
        raise CertsError(
            f"File has incorrect SHA256 hash.\n"
            f"Expected: {expected_hash}\n"
            f"Actual: {actual_hash}"
        )
