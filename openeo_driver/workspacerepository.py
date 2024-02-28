import abc

from openeo_driver.config import get_backend_config
from openeo_driver.workspace import Workspace


class WorkspaceRepository(abc.ABC):
    @abc.abstractmethod
    def get_by_id(self, workspace_id: str) -> Workspace:
        raise NotImplementedError


class BackendConfigWorkspaceRepository(WorkspaceRepository):
    def get_by_id(self, workspace_id: str) -> Workspace:
        return get_backend_config().workspaces[workspace_id]


backend_config_workspace_repository = BackendConfigWorkspaceRepository()
