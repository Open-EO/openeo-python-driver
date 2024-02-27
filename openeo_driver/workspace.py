import abc
import os.path
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
        merge = os.path.normpath(merge)
        subdirectory = merge[1:] if merge.startswith("/") else merge
        target_directory = self.root_directory / subdirectory
        target_directory.relative_to(self.root_directory)  # assert target_directory is in root_directory

        target_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, target_directory)
