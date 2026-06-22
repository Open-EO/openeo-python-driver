import json
import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict
from urllib.parse import urljoin, urlparse

import requests

from openeo_driver.datastructs import StacAsset
from openeo_driver.integrations.s3.client import S3ClientBuilder

_log = logging.getLogger(__name__)


def robust_urljoin(base: str, url: Optional[str], allow_fragments=True):
    """
    Overly-cautious wrapper around urljoin to allow joining s3-urls.
    """
    if not base.lower().startswith("s3://"):
        return urljoin(base, url, allow_fragments)

    # temporary switch to scheme that allows urljoin.
    base = base.replace("s3://", "file://")
    return_value = urljoin(base, url, allow_fragments)
    return return_value.replace("file://", "s3://")


# TODO: Check if pystac can natively loop over items/assets/files.


def _read_json(path: str) -> dict:
    if path.startswith("s3://"):
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path[1:]
        s3_instance = S3ClientBuilder.from_bucket(bucket)
        obj = s3_instance.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    elif path.startswith("http"):
        response = requests.get(path)
        response.raise_for_status()
        return response.json()
    else:
        assert os.path.exists(path), f"path does not exist: {path}"
        return json.loads(Path(path).read_text())


def find_stac_root(paths: Union[set, list], stac_root_filename: Optional[str] = "catalog.json") -> Optional[str]:
    paths = list(dict.fromkeys(paths))  # de-duplicate
    paths = [Path(p) for p in paths]

    def search(stac_root_filename_local: str):
        matches = [x for x in paths if x.name == stac_root_filename_local]
        if matches:
            if len(matches) > 1:
                _log.warning(f"Multiple STAC root files found: {[str(x) for x in matches]}. Using the first one.")
            return str(matches[0])
        return None

    if stac_root_filename:
        ret = search(stac_root_filename)
        if ret:
            return ret
    ret = search("catalog.json")
    if ret:
        return ret
    ret = search("catalogue.json")
    if ret:
        return ret
    ret = search("collection.json")
    if ret:
        return ret
    return None


def _make_relative(path: str, root_dir: str) -> str:
    root_dir = root_dir.rstrip("/") + "/"
    assert path.startswith(root_dir)
    return path[len(root_dir) :]


def get_files_from_stac_catalog(
    catalog_path: Union[str, Path], include_metadata=False, relative_paths: bool = False
) -> list:
    """
    Goes through the stac catalog recursively to find all files.
    """

    def recurse(item_path: Union[str, Path]) -> list:
        item_path: str = str(item_path)
        catalog_json = _read_json(item_path)

        all_files = []
        links = []
        if include_metadata:
            all_files.append(item_path)
        if "links" in catalog_json:
            links.extend(catalog_json["links"])
        if "assets" in catalog_json:
            links.extend(list(catalog_json["assets"].values()))
        for link in links:
            if "href" in link:
                href = link["href"]
                if href.startswith("file://"):
                    href = href[7:]
                href = robust_urljoin(item_path, href)

                if "rel" in link and (link["rel"] == "child" or link["rel"] == "item"):
                    all_files.extend(get_files_from_stac_catalog(href, include_metadata))
                else:
                    all_files.append(href)
        return all_files

    return_files = recurse(catalog_path)
    if relative_paths:
        # TODO: does this still works with recursive catalogs?
        root_dir = os.path.dirname(str(catalog_path))
        return_files = [_make_relative(f, root_dir) for f in return_files]

    return return_files


def get_assets_from_stac_catalog(catalog_path: Union[str, Path]) -> Dict[str, StacAsset]:
    catalog_path: str = str(catalog_path)
    catalog_json = _read_json(catalog_path)

    all_assets = {}
    links = []
    if "links" in catalog_json:
        links.extend(catalog_json["links"])
    if "assets" in catalog_json:
        links.extend(list(catalog_json["assets"].values()))
        assets = catalog_json["assets"]
        all_assets.update(assets)
    for link in links:
        if "href" in link:
            href = link["href"]
            if href.startswith("file://"):
                href = href[7:]
            href = robust_urljoin(catalog_path, href)

            if "rel" in link and (link["rel"] == "child" or link["rel"] == "item"):
                all_assets.update(get_assets_from_stac_catalog(href))
    return all_assets


def get_items_from_stac_catalog(catalog_path: Union[str, Path], make_hrefs_absolute=False) -> dict:
    catalog_path: str = str(catalog_path)
    catalog_json = _read_json(catalog_path)

    all_items = {}
    links = []
    if "links" in catalog_json:
        links.extend(catalog_json["links"])
    if "assets" in catalog_json:
        links.extend(list(catalog_json["assets"].values()))
        all_items.update({catalog_json["id"]: catalog_json})
    if make_hrefs_absolute:
        for item in all_items.values():
            if "assets" in item:
                for asset in item["assets"].values():
                    if "href" in asset:
                        asset["href"] = robust_urljoin(catalog_path, asset["href"])
    for link in links:
        if "href" in link:
            href = link["href"]
            if href.startswith("file://"):
                href = href[7:]
            href = robust_urljoin(catalog_path, href)

            if "rel" in link and (link["rel"] == "child" or link["rel"] == "item"):
                all_items.update(get_items_from_stac_catalog(href, make_hrefs_absolute))
    return all_items
