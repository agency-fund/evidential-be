from pathlib import Path

from xngin.apiserver.certs.certs import PATH_TO_AMAZON_TRUST_CA_BUNDLE


def test_certs_exist():
    assert Path(PATH_TO_AMAZON_TRUST_CA_BUNDLE).exists()
