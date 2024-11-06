import abc
import logging
import os.path
import shutil
from pathlib import Path


_log = logging.getLogger(__name__)


class Workspace(abc.ABC):
    @abc.abstractmethod
    def import_file(self, file: Path, merge: str, remove_original: bool = False) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def import_object(self, s3_uri: str, merge: str, remove_original: bool = False):
        raise NotImplementedError


class DiskWorkspace(Workspace):

    def __init__(self, root_directory: Path):
        self.root_directory = root_directory

    def import_file(self, file: Path, merge: str, remove_original: bool = False) -> str:
        merge = os.path.normpath(merge)
        subdirectory = merge[1:] if merge.startswith("/") else merge
        target_directory = self.root_directory / subdirectory
        target_directory.relative_to(self.root_directory)  # assert target_directory is in root_directory

        target_directory.mkdir(parents=True, exist_ok=True)

        operation = shutil.move if remove_original else shutil.copy
        operation(str(file), str(target_directory))

        _log.debug(f"{'moved' if remove_original else 'copied'} {file.absolute()} to {target_directory}")
        return f"file:{target_directory / file.name}"

    def import_object(self, s3_uri: str, merge: str, remove_original: bool = False):
        raise NotImplementedError(f"importing objects is not supported yet")
