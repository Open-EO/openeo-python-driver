import pytest

from openeo_driver.workspace import Workspace
from openeo_driver.workspacerepository import backend_config_workspace_repository


def test_backend_config_workspace_repository():
    workspace_repository = backend_config_workspace_repository
    workspace = workspace_repository.get_by_id("tmp")

    assert isinstance(workspace, Workspace)


def test_backend_config_workspace_repository_unknown_workspace():
    with pytest.raises(KeyError):
        workspace_repository = backend_config_workspace_repository
        workspace_repository.get_by_id("retteketet")
