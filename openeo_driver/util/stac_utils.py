import json
import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict
from urllib.parse import urljoin

import requests

from openeo_driver.datastructs import StacAsset

_log = logging.getLogger(__name__)


# TODO: Check if pystac can natively loop over items/assets/files.


def find_stac_root(paths: Union[set, list], stac_root_filename: Optional[str] = "catalog.json") -> Optional[str]:
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


def get_files_from_stac_catalog(catalog_path: Union[str, Path], include_metadata=False) -> list:
    """
    Goes through the stac catalog recursively to find all files.
    """
    if isinstance(catalog_path, str) and catalog_path.startswith("http"):
        response = requests.get(catalog_path)
        response.raise_for_status()
        catalog_json = response.json()
    else:
        catalog_path = str(catalog_path)
        assert os.path.exists(catalog_path)
        catalog_json = json.loads(Path(catalog_path).read_text())

    all_files = []
    links = []
    if include_metadata:
        all_files.append(catalog_path)
    if "links" in catalog_json:
        links.extend(catalog_json["links"])
    if "assets" in catalog_json:
        links.extend(list(catalog_json["assets"].values()))
    for link in links:
        if "href" in link:
            href = link["href"]
            if href.startswith("file://"):
                href = href[7:]
            href = urljoin(catalog_path, href)

            if "rel" in link and (link["rel"] == "child" or link["rel"] == "item"):
                all_files.extend(get_files_from_stac_catalog(href, include_metadata))
            else:
                all_files.append(href)
    return all_files


def get_assets_from_stac_catalog(catalog_path: Union[str, Path]) -> Dict[str, StacAsset]:
    if isinstance(catalog_path, str) and catalog_path.startswith("http"):
        response = requests.get(catalog_path)
        response.raise_for_status()
        catalog_json = response.json()
    else:
        catalog_path = str(catalog_path)
        assert os.path.exists(catalog_path)
        catalog_json = json.loads(Path(catalog_path).read_text())

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
            href = urljoin(catalog_path, href)

            if "rel" in link and (link["rel"] == "child" or link["rel"] == "item"):
                all_assets.update(get_assets_from_stac_catalog(href))
    return all_assets


def get_items_from_stac_catalog(catalog_path: Union[str, Path], make_hrefs_absolute=False) -> dict:
    if isinstance(catalog_path, str) and catalog_path.startswith("http"):
        response = requests.get(catalog_path)
        response.raise_for_status()
        catalog_json = response.json()
    else:
        catalog_path = str(catalog_path)
        assert os.path.exists(catalog_path), f"catalog_path does not exist: {catalog_path}"
        catalog_json = json.loads(Path(catalog_path).read_text())

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
                        asset["href"] = urljoin(catalog_path, asset["href"])
    for link in links:
        if "href" in link:
            href = link["href"]
            if href.startswith("file://"):
                href = href[7:]
            href = urljoin(catalog_path, href)

            if "rel" in link and (link["rel"] == "child" or link["rel"] == "item"):
                all_items.update(get_items_from_stac_catalog(href, make_hrefs_absolute))
    return all_items
