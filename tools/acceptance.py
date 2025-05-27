import json
import tempfile

import pytest
import requests

from xngin.apiserver.testing.assertions import assert_same


@pytest.fixture
def pyhost(request):
    """Python implementation."""
    return request.config.getoption("--p", "http://localhost:8000")


@pytest.fixture
def rhost(request):
    """R implementation."""
    return request.config.getoption("--r", "http://localhost:8383")


@pytest.fixture
def data_dir(request):
    """R implementation."""
    return request.config.getoption("--data-dir", "acceptance_data/")


@pytest.fixture
def alt_payload_path(request):
    """Override the parameterized tests with this payload file path."""
    file = request.config.getoption("--req")
    if file == "":
        return ""
    with open(file) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def client():
    session = requests.Session()
    session.headers.update({
        "accept": "application/json",
        "Content-Type": "application/json",
        # "Config-ID": "customer",
        "Datasource-ID": "customer",
    })
    yield session


def get_test(
    client,
    pyhost,
    rhost,
    endpoint,
    prefix,
    payload_path: str | None = None,
    alt_payload_path: str | None = None,
):
    body = alt_payload_path
    if not body and payload_path:
        with open(payload_path) as f:
            body = json.load(f)

    def call(host, file_prefix):
        if not host:
            return None, None

        url = f"{host}/{endpoint}"
        print(f"{url} file={payload_path}\n\t{body}")
        resp = client.post(url=url, json=body) if body else client.get(url=url)
        assert resp.status_code == 200, host

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, prefix=f"{file_prefix}.", suffix=".json"
        ) as tmpf:
            json.dump(resp.json(), tmpf, indent=2)
            return resp, tmpf

    presp, t1 = call(pyhost, f"{prefix}.p")
    rresp, t2 = call(rhost, f"{prefix}.p")

    # assert presp.json() == rresp.json()
    if presp and rresp:
        assert_same(
            presp.json(),
            rresp.json(),
            # deepdiff_kwargs=deepdiff_kwargs,
            extra=f"vimdiff {t1.name}  {t2.name}",
        )


def test_filters(client, pyhost, rhost):
    get_test(client, pyhost, rhost, "filters?participant_type=groups", "filters")


def test_metrics(client, pyhost, rhost):
    get_test(client, pyhost, rhost, "metrics?participant_type=groups", "metrics")


def test_strata(client, pyhost, rhost):
    get_test(client, pyhost, rhost, "strata?participant_type=groups", "strata")


@pytest.mark.parametrize("payload_file", ["power1"])
def test_power(client, pyhost, rhost, data_dir, payload_file, alt_payload_path):
    payload_path = f"{data_dir}{payload_file}"
    get_test(
        client,
        pyhost,
        rhost,
        "power?refresh=false",
        "power",
        payload_path,
        alt_payload_path,
    )


@pytest.mark.parametrize("payload_file", ["assign1", "assign2"])
def test_assign(client, pyhost, rhost, data_dir, payload_file, alt_payload_path):
    payload_path = f"{data_dir}{payload_file}"
    get_test(
        client,
        pyhost,
        rhost,
        "assign?chosen_n=100&random_state=0&refresh=false",
        "assign",
        payload_path,
        alt_payload_path,
    )


@pytest.mark.parametrize("payload_file", ["commit"])
def test_commit(client, pyhost, rhost, data_dir, payload_file, alt_payload_path):
    payload_path = f"{data_dir}{payload_file}"
    get_test(
        client,
        pyhost,
        rhost,
        "commit?user_id=commituser",
        "commit",
        payload_path,
        alt_payload_path,
    )
