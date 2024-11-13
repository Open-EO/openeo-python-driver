import abc
import logging
import os.path
import shutil
from pathlib import Path, PurePath

from pystac import Collection, STACObject
from pystac.catalog import CatalogType
from pystac.layout import TemplateLayoutStrategy

_log = logging.getLogger(__name__)


class Workspace(abc.ABC):
    @abc.abstractmethod
    def import_file(self, file: Path, merge: str, remove_original: bool = False) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def import_object(self, s3_uri: str, merge: str, remove_original: bool = False):
        raise NotImplementedError

    @abc.abstractmethod
    def merge_files(self, stac_resource: Collection, target: PurePath, remove_original: bool = False):
        # TODO: use an abstraction like a dict instead?
        # TODO: is a PurePath object fine as an abstraction?
        raise NotImplementedError


class DiskWorkspace(Workspace):

    def __init__(self, root_directory: Path):
        self.root_directory = root_directory

    def import_file(self, file: Path, merge: str, remove_original: bool = False) -> str:
        merge = os.path.normpath(merge)

        # merge points to a file with the STAC Collection; all STAC documents and assets end up in its parent directory
        subdirectory = Path(merge[1:] if merge.startswith("/") else merge).parent
        target_directory = self.root_directory / subdirectory
        target_directory.relative_to(self.root_directory)  # assert target_directory is in root_directory

        target_directory.mkdir(parents=True, exist_ok=True)

        operation = shutil.move if remove_original else shutil.copy
        operation(str(file), str(target_directory))

        _log.debug(f"{'moved' if remove_original else 'copied'} {file.absolute()} to {target_directory}")
        return f"file:{target_directory / file.name}"

    def import_object(self, s3_uri: str, merge: str, remove_original: bool = False):
        raise NotImplementedError(f"importing objects is not supported yet")

    def merge_files(self, stac_resource: STACObject, target: PurePath, remove_original: bool = False):
        # FIXME: merge new $stac_resource with the one in $this_workspace at $target
        # FIXME: export STAC resources and assets underneath as well
        # FIXME: support remove_original and return equivalent workspace URIs (pass alternate_key and put workspace URIs in "alternate"?)
        target = os.path.normpath(target)
        target = Path(target[1:] if target.startswith("/") else target)
        target = self.root_directory / target
        target.relative_to(self.root_directory)  # assert target_directory is in root_directory

        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(stac_resource, Collection):
            # collection ends up in file $target
            # items and assets and up in directory $target.parent
            for item in stac_resource.get_items(recursive=True):
                for asset in item.assets.values():
                    shutil.copy(asset.href, target.parent)
                    asset.href = Path(asset.href).name

            stac_resource.normalize_and_save(
                root_href=str(target.parent),
                catalog_type=CatalogType.SELF_CONTAINED,
                strategy=TemplateLayoutStrategy(collection_template=target.name, item_template="${id}.json"),
            )
        else:
            raise NotImplementedError(stac_resource)
