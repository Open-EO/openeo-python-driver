import abc
import shutil
from abc import ABC
from pathlib import Path


class Workspace(ABC):
    @abc.abstractmethod
    def import_file(self, file: Path, merge: str):
        raise NotImplementedError


class DiskWorkspace(Workspace):
    def __init__(self, root_directory: Path):
        self.root_directory = root_directory

    def import_file(self,
                    file: Path,
                    merge: str):
        # TODO: create missing directories?
        shutil.copy(file, self.root_directory / merge)
