import abc
import logging
import os.path
import shutil
from pathlib import Path


_log = logging.getLogger(__name__)


class Workspace(abc.ABC):
    @abc.abstractmethod
    def import_file(self, file: Path, merge: str):
        raise NotImplementedError

    @abc.abstractmethod
    def import_object(self, s3_uri: str, merge: str):
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

        _log.debug(f"copied {file.absolute()} to {target_directory}")

    def import_object(self, s3_uri: str, merge: str):
        super().import_object(s3_uri, merge)
