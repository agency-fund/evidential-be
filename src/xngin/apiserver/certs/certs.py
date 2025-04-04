from pathlib import Path

PATH_TO_AMAZON_TRUST_CA_BUNDLE = (
    (Path(__file__).parent / "amazon-trust-ca-bundle.crt")
    .resolve(strict=True)
    .as_posix()
)
