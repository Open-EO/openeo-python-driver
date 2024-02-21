import abc
import shutil
from abc import ABC
from pathlib import Path


class Workspace(ABC):
    @abc.abstractmethod
    def write(self, source_file: str, merge: str):
        raise NotImplementedError


class DiskWorkspace(Workspace):
    def __init__(self, root_directory: Path):
        self.root_directory = root_directory

    def write(self,
              source_file: str,  # TODO: assumes source files are on disk
              merge: str):
        # TODO: create missing directories?
        shutil.copy(source_file, self.root_directory / merge)
