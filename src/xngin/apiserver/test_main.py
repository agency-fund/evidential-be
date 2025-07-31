import os
import tempfile
from pathlib import Path

import pytest

from xngin.apiserver.conftest import (
    DeveloperErrorRunFromRootOfRepositoryPleaseError,
    ensure_correct_working_directory_impl,
)
from xngin.apiserver.settings import (
    CannotFindParticipantsError,
)


def test_root_get_api(client):
    response = client.get("/")
    assert response.status_code == 404


def test_static_settings_contains_testing_datasource(static_settings):
    assert static_settings.get_datasource("testing").config.dwh.user == "postgres"


def test_find_participants_raises_specific_exception_for_undefined_participants(
    static_settings,
):
    config = static_settings.get_datasource("testing").config
    with pytest.raises(CannotFindParticipantsError):
        config.find_participants("bad_type")


def test_ensure_correct_working_directory_impl_outside_home_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # Raises since current dir is not repo root
        with pytest.raises(DeveloperErrorRunFromRootOfRepositoryPleaseError):
            ensure_correct_working_directory_impl()

        # Passes since current dir appears to be repo root
        file_path = os.path.join(tmpdir, "pyproject.toml")
        with open(file_path, "w", encoding="utf-8"):
            pass  # Just touch the file to make it a valid repository
        ensure_correct_working_directory_impl()

        # Starting one level deeper should NOT pass since we're not under the home directory and it
        # might not be safe to walk up the path.
        child_dir = os.path.join(tmpdir, "child")
        os.mkdir(child_dir)
        os.chdir(child_dir)
        with pytest.raises(DeveloperErrorRunFromRootOfRepositoryPleaseError):
            ensure_correct_working_directory_impl()


def test_ensure_correct_working_directory_impl_below_home_dir():
    with tempfile.TemporaryDirectory(dir=Path.home()) as tmpdir:
        os.chdir(tmpdir)

        # Raises since we're not the project root.
        with pytest.raises(DeveloperErrorRunFromRootOfRepositoryPleaseError):
            ensure_correct_working_directory_impl()

        # Passes since current dir appears to be repo root
        file_path = os.path.join(tmpdir, "pyproject.toml")
        with open(file_path, "w", encoding="utf-8"):
            pass  # Just touch the file to make it a valid repository
        ensure_correct_working_directory_impl()

        # Starting one level deeper should pass.
        child_dir = os.path.join(tmpdir, "child")
        os.mkdir(child_dir)
        os.chdir(child_dir)
        ensure_correct_working_directory_impl()

        # But now that pyproject.toml doesn't exist above us, we fail.
        os.remove(file_path)
        with pytest.raises(DeveloperErrorRunFromRootOfRepositoryPleaseError):
            ensure_correct_working_directory_impl()
