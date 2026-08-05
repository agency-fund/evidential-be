from unittest import mock

import pytest
import typer
from google.api_core.exceptions import Forbidden, NotFound
from google.cloud import bigquery

from xngin.cli.main import bigquery_dataset_delete


def _patched_client(side_effect=None):
    """Patches bigquery.Client so delete_dataset() raises side_effect, or succeeds when it is None."""
    client = mock.MagicMock()
    client.delete_dataset.side_effect = side_effect
    return mock.patch.object(bigquery, "Client", return_value=client)


def test_bigquery_dataset_delete_succeeds_when_the_dataset_was_deleted():
    with _patched_client():
        bigquery_dataset_delete(project_id="p", dataset_id="d")


def test_bigquery_dataset_delete_succeeds_when_the_dataset_is_absent():
    """CI cleanup runs even when the job failed before the dataset was created."""
    with _patched_client(NotFound("no such dataset")):
        bigquery_dataset_delete(project_id="p", dataset_id="d")


def test_bigquery_dataset_delete_fails_on_other_api_errors():
    with _patched_client(Forbidden("caller lacks bigquery.datasets.delete")), pytest.raises(typer.Exit) as excinfo:
        bigquery_dataset_delete(project_id="p", dataset_id="d")
    assert excinfo.value.exit_code == 1
