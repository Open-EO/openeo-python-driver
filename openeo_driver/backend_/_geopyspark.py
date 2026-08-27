"""

Note: this module is somewhat of a temporary, stop-gap solution
for the following considerations:
- Extract the existing, but openeo-geopyspark-driver oriented
  batch job result metadata handling from the generic view layer.
- Allow openeo-aggregator to inject an alternative implementation
  of the batch job result metadata handling, without having
  to wrestle geopyspark-driver specific quirks
  in the openeo-python-driver view layer.
- It is still kept in openeo-python-driver for now to ease the migration path.
  Ideally, in the long term however, most of this should probably
  be moved to openeo-geopyspark-driver.

"""


import copy
import logging
import os
import urllib.parse
from pathlib import Path
from typing import List, Optional

import flask
import openeo
import openeo.metadata
from openeo.util import Rfc3339, TimingLogger, deep_get, dict_no_none, rfc3339
from openeo.utils.version import ComparableVersion

from openeo_driver.backend import (
    BatchJobMetadata,
    BatchJobResultMetadata,
    BatchJobs,
)
from openeo_driver.views_.batch_jobs import list_job_results_add_basic_links
from openeo_driver.config import get_backend_config
from openeo_driver.constants import (
    ITEM_LINK_PROPERTY,
    JOB_STATUS,
    STAC_EXTENSION,
    STAC_ITEM_MEDIA_TYPE,
)
from openeo_driver.datacube import DriverMlModel
from openeo_driver.errors import FilePathInvalidException, JobNotFinishedException, OpenEOApiException
from openeo_driver.jobregistry import PARTIAL_JOB_STATUS
from openeo_driver.users import user_id_b64_encode
from openeo_driver.util.geometry import BoundingBox
from openeo_driver.util.stac import sniff_stac_extension_prefix


_log = logging.getLogger(__name__)


def list_job_results(
    *, batch_jobs: BatchJobs, job_id: str, user_id: str, partial: bool = False, api_version: ComparableVersion
) -> dict:
    # TODO: this is the legacy openeo-geopyspark-driver oriented implementation

    with TimingLogger(f"backend_implementation.batch_jobs.get_job_info({job_id=}, {user_id=})", logger=_log):
        job_info = batch_jobs.get_job_info(job_id, user_id)

    if job_info.status != JOB_STATUS.FINISHED:
        if not partial:
            raise JobNotFinishedException()
        else:
            return _list_job_results_partial(user_id=user_id, job_id=job_id, job_info=job_info, partial=partial)

    with TimingLogger(f"backend_implementation.batch_jobs.get_result_metadata({job_id=}, {user_id=})", logger=_log):
        result_metadata = batch_jobs.get_result_metadata(job_id=job_id, user_id=user_id)

    if api_version.at_least("1.1.0"):
        if result_metadata.items:
            # "STAC 1.1" style result listing (STAC Collection with focus on item-level assets)
            return _list_job_results_stac11(
                user_id=user_id, job_id=job_id, job_info=job_info, result_metadata=result_metadata
            )
        else:
            # "openEO 1.1.0" style result listing (STAC Collection with focus on collection-level assets)
            return _list_job_results_openeo110(
                user_id=user_id, job_id=job_id, job_info=job_info, result_metadata=result_metadata
            )
    else:
        # "openEO 1.0.0" style result listing (STAC Item)
        _log.warning(f"Using old STAC Item style job result listing for {job_id=} ({api_version=})")
        return _list_job_results_openeo100(
            user_id=user_id, job_id=job_id, job_info=job_info, result_metadata=result_metadata
        )


def _list_job_results_partial(*, user_id: str, job_id: str, job_info: BatchJobMetadata, partial: bool) -> dict:
    links = list_job_results_add_basic_links(
        links=[], job_id=job_id, user_id=user_id, partial=partial, add_card4l=False
    )
    result = {
        "openeo:status": PARTIAL_JOB_STATUS.for_job_status(job_info.status),
        "type": "Collection",
        "stac_version": "1.0.0",
        "id": job_id,
        "title": job_info.title or f"Unfinished batch job {job_id}",
        "description": job_info.description or f"Results for batch job {job_id}",
        "license": "proprietary",  # TODO?
        "extent": {
            "spatial": {"bbox": [[-180, -90, 180, 90]]},
            "temporal": {"interval": [[rfc3339.now_utc(), rfc3339.now_utc()]]},
        },
        "links": links,
    }
    return result






def _job_result_item_url(*, job_id: str, item_id: str, user_id: str, is11: bool = False) -> str:
    signer = get_backend_config().url_signer

    method_start = ".get_job_result_item"
    if is11:
        method_start = method_start + "11"
    if not signer:
        return flask.url_for(method_start, job_id=job_id, item_id=item_id, _external=True)

    expires = signer.get_expires()
    secure_key = signer.sign_job_item(job_id=job_id, user_id=user_id, item_id=item_id, expires=expires)
    user_base64 = user_id_b64_encode(user_id)
    return flask.url_for(
        method_start + "_signed",
        job_id=job_id,
        user_base64=user_base64,
        secure_key=secure_key,
        item_id=item_id,
        expires=expires,
        _external=True,
    )


def _list_job_results_stac11(
    *,
    user_id: str,
    job_id: str,
    job_info: BatchJobMetadata,
    result_metadata: BatchJobResultMetadata,
) -> dict:
    """
    Batch job result listing in "STAC1.1" style:
    a STAC collection, collection-level assets are deprecated in favor of item-level assets,
    asset keys should not be assumed to be filenames
    """
    to_datetime = Rfc3339(propagate_none=True).datetime

    links: List[dict] = copy.deepcopy(result_metadata.links or job_info.links or [])
    links = list_job_results_add_basic_links(links=links, job_id=job_id, user_id=user_id)

    def intersect_band_array(list1, list2):
        band_result = []
        for item1 in list1:
            if isinstance(item1, dict) and "name" in item1:
                for item2 in list2:
                    if isinstance(item1, dict) and "name" in item1 and item1["name"] == item2["name"]:
                        band_result.append(intersect_dicts(item1, item2))
        return band_result

    def intersect_dicts(dict1, dict2):
        result = {}
        for key in dict1:
            if key in dict2:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    # Recursively intersect nested dictionaries
                    nested_result = intersect_dicts(dict1[key], dict2[key])
                    if nested_result:  # Only add if the nested result is not empty
                        result[key] = nested_result
                elif isinstance(dict1[key], list) and isinstance(dict2[key], list) and key == "bands":
                    result[key] = intersect_band_array(dict1[key], dict2[key])
                elif dict1[key] == dict2[key]:
                    # Retain the key-value pair if values are equal
                    result[key] = dict1[key]
        return result

    item_assets = {}
    assets = {}
    for item_key, item_metadata in result_metadata.items.items():
        for asset_key, asset_metadata in item_metadata.get("assets", {}).items():
            if "output_dir" in asset_metadata:
                out_dir = asset_metadata.get("output_dir")
                _log.info(f"asset has output dir {out_dir} and href {asset_metadata.get('href')}")
                common = os.path.commonpath([asset_metadata.get("href"), out_dir])
                href = os.path.relpath(asset_metadata.get("href"), common)
            else:
                href = asset_metadata.get("href")
            asset_object = _asset_object(
                job_id=job_id,
                user_id=user_id,
                filename=href,
                asset_metadata=asset_metadata,
                job_info=job_info,
                stac11=True,
            )
            assets[item_key + "_" + asset_key] = asset_object
            item_asset = dict_no_none(
                {
                    "type": asset_object.get("type"),
                    "roles": asset_object.get("roles"),
                    "bands": asset_object.get("bands"),
                    "proj:bbox": asset_object.get("proj:bbox"),
                    "proj:epsg": asset_object.get("proj:epsg"),
                    "proj:shape": asset_object.get("proj:shape"),
                    "file:size": asset_object.get("file:size"),
                }
            )
            if asset_key not in item_assets:
                item_assets[asset_key] = item_asset
            else:
                item_assets[asset_key] = intersect_dicts(item_assets[asset_key], item_asset)
    for item_id in result_metadata.items.keys():
        links.append(
            {
                "rel": "item",
                "href": _job_result_item_url(job_id=job_id, item_id=item_id, user_id=user_id, is11=True),
                "type": STAC_ITEM_MEDIA_TYPE,
            }
        )
    stac_version = "1.1.0"

    links = [_normalize_job_result_link(link=k, job_id=job_id, user_id=user_id) for k in links]

    result = dict_no_none(
        {
            "type": "Collection",
            "stac_version": stac_version,
            "stac_extensions": [
                STAC_EXTENSION.EO_V110,
                STAC_EXTENSION.FILEINFO,
                STAC_EXTENSION.PROCESSING,
                STAC_EXTENSION.PROJECTION_V120,
            ],
            "id": job_id,
            "title": job_info.title,
            "description": job_info.description or f"Results for batch job {job_id}",
            "license": "proprietary",  # TODO?
            "extent": {
                "spatial": {"bbox": [job_info.bbox] if job_info.bbox else [[-180, -90, 180, 90]]},
                "temporal": {"interval": [[to_datetime(job_info.start_datetime), to_datetime(job_info.end_datetime)]]},
            },
            "summaries": {"instruments": job_info.instruments} if job_info.instruments else {},
            "providers": result_metadata.providers or None,
            "links": links,
            "assets": assets,
            "item_assets": item_assets,
            "openeo:status": PARTIAL_JOB_STATUS.FINISHED,
        }
    )
    return result


def _list_job_results_openeo110(
    *,
    user_id: str,
    job_id: str,
    job_info: BatchJobMetadata,
    result_metadata: BatchJobResultMetadata,
) -> dict:
    """
    Batch job result listing in "openEO API 1.1.0, but pre-STAC1.1" style:
    a STAC collection, but with focus on collection-level assets
    (with filenames as asset keys)
    """
    to_datetime = Rfc3339(propagate_none=True).datetime
    ml_model_metadata = None

    links: List[dict] = copy.deepcopy(result_metadata.links or job_info.links or [])
    links = list_job_results_add_basic_links(links=links, job_id=job_id, user_id=user_id)

    assets = {
        filename: _asset_object(
            job_id=job_id,
            user_id=user_id,
            filename=filename,
            asset_metadata=asset_metadata,
            job_info=job_info,
            stac11=False,
        )
        for filename, asset_metadata in result_metadata.assets.items()
        if asset_metadata.get("asset", True)
    }

    item_assets = None
    for filename, metadata in result_metadata.assets.items():
        if "data" in metadata.get("roles", []) and any(
            media_type in metadata.get("type", "")
            for media_type in ["geotiff", "netcdf", "text/csv", "application/parquet"]
        ):
            links.append(
                {
                    "rel": "item",
                    "href": _job_result_item_url(job_id=job_id, item_id=filename, user_id=user_id),
                    "type": STAC_ITEM_MEDIA_TYPE,
                }
            )
        elif metadata.get("ml_model_metadata", False):
            # TODO: Currently we only support one ml_model per batch job.
            ml_model_metadata = metadata
            links.append(
                {
                    "rel": "item",
                    "href": _job_result_item_url(job_id=job_id, item_id=filename, user_id=user_id),
                    "type": "application/json",
                }
            )
    stac_version = "1.0.0"

    links = [_normalize_job_result_link(link=k, job_id=job_id, user_id=user_id) for k in links]

    result = dict_no_none(
        {
            "type": "Collection",
            "stac_version": stac_version,
            "stac_extensions": [
                STAC_EXTENSION.EO_V110,
                STAC_EXTENSION.FILEINFO,
                STAC_EXTENSION.PROCESSING,
                STAC_EXTENSION.PROJECTION_V120,
            ],
            "id": job_id,
            "title": job_info.title,
            "description": job_info.description or f"Results for batch job {job_id}",
            "license": "proprietary",  # TODO?
            "extent": {
                "spatial": {"bbox": [job_info.bbox] if job_info.bbox else [[-180, -90, 180, 90]]},
                "temporal": {"interval": [[to_datetime(job_info.start_datetime), to_datetime(job_info.end_datetime)]]},
            },
            "summaries": {"instruments": job_info.instruments} if job_info.instruments else {},
            "providers": result_metadata.providers or None,
            "links": links,
            "assets": assets,
            "item_assets": item_assets,
            "openeo:status": PARTIAL_JOB_STATUS.FINISHED,
        }
    )

    if ml_model_metadata is not None:
        result["stac_extensions"].extend(ml_model_metadata.get("stac_extensions", []))
        if "summaries" not in result.keys():
            result["summaries"] = {}
        if "properties" in ml_model_metadata.keys():
            ml_model_properties = ml_model_metadata["properties"]
            learning_approach = ml_model_properties.get("ml-model:learning_approach", None)
            prediction_type = ml_model_properties.get("ml-model:prediction_type", None)
            architecture = ml_model_properties.get("ml-model:architecture", None)
            result["summaries"].update(
                {
                    "ml-model:learning_approach": [learning_approach] if learning_approach is not None else [],
                    "ml-model:prediction_type": [prediction_type] if prediction_type is not None else [],
                    "ml-model:architecture": [architecture] if architecture is not None else [],
                }
            )
    return result


def _list_job_results_openeo100(
    *,
    user_id: str,
    job_id: str,
    job_info: BatchJobMetadata,
    result_metadata: BatchJobResultMetadata,
) -> dict:
    """
    Batch job result listing in deprecated "openEO API 1.0.0" style:
    a STAC Item (type "Feature")
    """

    links: List[dict] = copy.deepcopy(result_metadata.links or job_info.links or [])
    links = list_job_results_add_basic_links(links=links, job_id=job_id, user_id=user_id)

    assets = {
        filename: _asset_object(
            job_id=job_id,
            user_id=user_id,
            filename=filename,
            asset_metadata=asset_metadata,
            job_info=job_info,
            stac11=False,
        )
        for filename, asset_metadata in result_metadata.assets.items()
        if asset_metadata.get("asset", True)
    }

    result = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": job_info.id,
        "properties": _properties_from_job_info(job_info),
        "assets": assets,
        "links": links,
        "openeo:status": PARTIAL_JOB_STATUS.FINISHED,
    }
    if result_metadata.providers:
        result["providers"] = result_metadata.providers

    geometry = job_info.geometry
    result["geometry"] = geometry
    if geometry:
        result["bbox"] = job_info.bbox

    result["stac_extensions"] = [
        STAC_EXTENSION.PROCESSING,
        STAC_EXTENSION.CARD4LOPTICAL,
        STAC_EXTENSION.FILEINFO,
    ]

    if sniff_stac_extension_prefix(result["assets"].values(), prefix="eo:"):
        result["stac_extensions"].append(STAC_EXTENSION.EO_V110)

    if any(key.startswith("proj:") for key in result["properties"]) or any(
        key.startswith("proj:") for key in result["assets"]
    ):
        result["stac_extensions"].append(STAC_EXTENSION.PROJECTION_V120)

    return result


def _asset_object(
    job_id, user_id, filename: str, asset_metadata: dict, job_info: BatchJobMetadata, stac11: bool
) -> dict:
    result_dict = dict_no_none(
        {
            "title": asset_metadata.get("title", filename),
            "href": asset_metadata.get(BatchJobs.ASSET_PUBLIC_HREF)
            or get_backend_config().asset_url.build_url(
                asset_metadata=asset_metadata, asset_name=filename, job_id=job_id, user_id=user_id
            ),
            "type": asset_metadata.get("type", asset_metadata.get("media_type", "application/octet-stream")),
            "roles": asset_metadata.get("roles", ["data"]),
            # TODO: eliminate this legacy "raster:bands" construct at some point?
            "raster:bands": None if stac11 else asset_metadata.get("raster:bands"),
            "file:size": asset_metadata.get("file:size"),
            "alternate": asset_metadata.get("alternate"),
        }
    )
    if filename.endswith(".model"):
        # Machine learning models.
        return result_dict
    bands = asset_metadata.get("bands")

    if bands:
        # TODO: #298 this is a quick stop-gap solution for lack of clear API
        #       what "bands" actually is expected to be:
        #       a list of Band objects (current approach in openeo-geopyspark-driver)
        #       or a list of dictionaries (as handled in openeo-aggregator)
        # TODO: move this normalization to a more general utility?
        bands = [
            openeo.metadata.Band(
                name=b.get("name"),
                common_name=b.get("eo:common_name") or b.get("common_name"),
                wavelength_um=b.get("eo:center_wavelength") or b.get("center_wavelength"),
            )
            if isinstance(b, dict)
            else b
            for b in bands
        ]

        # TODO: eliminate this legacy "eo:bands" construct at some point?
        if not stac11:
            result_dict["eo:bands"] = [
                dict_no_none(
                    {
                        "name": band.name,
                        "common_name": band.common_name,
                        "center_wavelength": band.wavelength_um,
                    }
                )
                for band in bands
            ]
        else:

            def raster_bands(band_index) -> dict:
                rb = asset_metadata.get("raster:bands", [])
                return rb[band_index] if band_index < len(rb) else {}

            result_dict["bands"] = [
                dict_no_none(
                    {
                        **{
                            "name": band.name,
                            "eo:common_name": band.common_name,
                            "eo:center_wavelength": band.wavelength_um,
                        },
                        **raster_bands(i),
                    }
                )
                for (i, band) in enumerate(bands)
            ]

    asset_proj_epsg = asset_metadata.get("proj:epsg", job_info.epsg)
    result_dict.update(
        dict_no_none(
            {
                "proj:bbox": asset_metadata.get("proj:bbox", job_info.proj_bbox),
                "proj:epsg": asset_proj_epsg,
                "proj:code": f"EPSG:{asset_proj_epsg}" if asset_proj_epsg else None,
                "proj:shape": asset_metadata.get("proj:shape", job_info.proj_shape),
            }
        )
    )

    if "file:size" not in result_dict and "output_dir" in asset_metadata:
        the_file = Path(asset_metadata["output_dir"]) / filename
        if the_file.exists():
            size_in_bytes = the_file.stat().st_size
            result_dict["file:size"] = size_in_bytes

    return result_dict


def _properties_from_job_info(job_info: BatchJobMetadata) -> dict:
    to_datetime = Rfc3339(propagate_none=True).datetime

    properties = dict_no_none(
        {
            "title": job_info.title,
            "description": job_info.description,
            "created": to_datetime(job_info.created),
            "updated": to_datetime(job_info.updated),
            "card4l:specification": "SR",
            "card4l:specification_version": "5.0",
            "processing:facility": get_backend_config().processing_facility,
            "processing:software": get_backend_config().processing_software,
        }
    )
    properties["datetime"] = None

    start_datetime = to_datetime(job_info.start_datetime)
    end_datetime = to_datetime(job_info.end_datetime)

    if start_datetime == end_datetime:
        properties["datetime"] = start_datetime
    else:
        if start_datetime:
            properties["start_datetime"] = start_datetime
        if end_datetime:
            properties["end_datetime"] = end_datetime

    if job_info.instruments:
        properties["instruments"] = job_info.instruments

    if job_info.epsg:
        properties["proj:epsg"] = job_info.epsg
        properties["proj:code"] = f"EPSG:{job_info.epsg}"

    if job_info.proj_bbox:
        properties["proj:bbox"] = job_info.proj_bbox

    if job_info.proj_shape:
        properties["proj:shape"] = job_info.proj_shape

    properties["card4l:processing_chain"] = job_info.process

    return properties


def _normalize_job_result_link(link: dict, *, job_id: str, user_id: str) -> dict:
    if link.get(ITEM_LINK_PROPERTY.EXPOSE_AUXILIARY, False):
        link = _auxiliary_link(exposable_link=link, job_id=job_id, user_id=user_id)

    if link.get("rel") == "original":
        # TODO: Cleanup
        # TODO: this "original" handling is highly specific to a niche openeo-geopyspark-driver feature (CWL)
        #       and does not really fit the generic nature of openeo-python-driver.
        #       Can this be generalized more cleanly? Or moved to openeo-geopyspark-driver?
        try:
            # TODO: assumes file is not nested
            asset_name = urllib.parse.urlparse(link["href"]).path.split("/")[-1]
            href = flask.url_for(
                ".download_job_result",
                job_id=job_id,
                filename=asset_name,
                _external=True,
            )
            link = dict(**link, href=href)
        except Exception as e:
            _log.warning("Error when making URL for 'original' link: " + str(e))

    return link


def _auxiliary_link(exposable_link: dict, *, job_id: str, user_id: str) -> dict:
    auxiliary_filename = urllib.parse.urlparse(exposable_link["href"]).path.split("/")[
        -1
    ]  # TODO: assumes file is not nested

    if exposable_link["href"].startswith("s3://"):
        # TODO: asset.build_url is made for assets, but not aux links, right?
        href = get_backend_config().asset_url.build_url(
            asset_metadata={"href": exposable_link["href"]},  # TODO: clean up this hack to support s3proxy
            asset_name=auxiliary_filename,
            job_id=job_id,
            user_id=user_id,
        )
    else:
        signer = get_backend_config().url_signer
        if signer:
            expires = signer.get_expires()
            secure_key = signer.sign_job_asset(
                job_id=job_id, user_id=user_id, filename=auxiliary_filename, expires=expires
            )
            user_base64 = user_id_b64_encode(user_id)
            href = flask.url_for(
                ".download_job_auxiliary_file_signed",
                job_id=job_id,
                user_base64=user_base64,
                filename=auxiliary_filename,
                expires=expires,
                secure_key=secure_key,
                _external=True,
            )
        else:
            href = flask.url_for(
                ".download_job_auxiliary_file", job_id=job_id, filename=auxiliary_filename, _external=True
            )

    return dict_no_none(
        href=href,
        rel=exposable_link.get("rel"),
        type=exposable_link.get("type"),
    )


def get_item_metadata_doc(
    *, batch_jobs: BatchJobs, job_id: str, item_id: str, user_id: str, format: Optional[str] = None
) -> dict:
    if format == "stac11":
        return _get_job_result_item11(batch_jobs=batch_jobs, job_id=job_id, item_id=item_id, user_id=user_id)
    else:
        return _get_job_result_item(batch_jobs=batch_jobs, job_id=job_id, item_id=item_id, user_id=user_id)


def _get_job_result_item(*, batch_jobs: BatchJobs, job_id: str, item_id: str, user_id: str) -> dict:
    if item_id == DriverMlModel.METADATA_FILE_NAME:
        return _download_ml_model_metadata(batch_jobs=batch_jobs, job_id=job_id, file_name=item_id, user_id=user_id)

    results = batch_jobs.get_result_assets(job_id=job_id, user_id=user_id)

    assets_for_item_id = {
        asset_filename: metadata for asset_filename, metadata in results.items() if asset_filename.startswith(item_id)
    }

    if len(assets_for_item_id) != 1:
        raise AssertionError(f"expected exactly 1 asset with file name {item_id}. Got {len(assets_for_item_id)}")

    asset_filename, asset_metadata = next(iter(assets_for_item_id.items()))

    job_info = batch_jobs.get_job_info(job_id, user_id)

    properties = {"datetime": asset_metadata.get("datetime")}
    if properties["datetime"] is None:
        to_datetime = Rfc3339(propagate_none=True).datetime

        start_datetime = asset_metadata.get("start_datetime") or to_datetime(job_info.start_datetime)
        end_datetime = asset_metadata.get("end_datetime") or to_datetime(job_info.end_datetime)

        if start_datetime == end_datetime:
            properties["datetime"] = start_datetime
        else:
            if start_datetime:
                properties["start_datetime"] = start_datetime
            if end_datetime:
                properties["end_datetime"] = end_datetime

    if job_info.proj_shape:
        properties["proj:shape"] = job_info.proj_shape
    if job_info.proj_bbox:
        properties["proj:bbox"] = job_info.proj_bbox
    if job_info.epsg:
        properties["proj:epsg"] = job_info.epsg
        properties["proj:code"] = f"EPSG:{job_info.epsg}"

    bbox = asset_metadata.get("bbox", job_info.bbox)
    if not bbox and job_info.proj_bbox and job_info.epsg:
        bbox = BoundingBox.from_wsen_tuple(job_info.proj_bbox, crs=job_info.epsg).reproject(4326).as_wsen_tuple()
    geometry = asset_metadata.get("geometry", job_info.geometry)
    if not geometry and job_info.proj_bbox and job_info.epsg:
        geometry = BoundingBox.from_wsen_tuple(wsen=job_info.proj_bbox, crs=job_info.epsg).as_geojson()

    stac_item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            STAC_EXTENSION.EO_V110,
            STAC_EXTENSION.FILEINFO,
            STAC_EXTENSION.PROJECTION_V120,
        ],
        "id": item_id,
        "geometry": geometry,
        "bbox": bbox,
        "properties": properties,
        "links": [
            {
                "rel": "self",
                # MUST be absolute
                "href": flask.url_for(".get_job_result_item", job_id=job_id, item_id=item_id, _external=True),
                "type": STAC_ITEM_MEDIA_TYPE,
            },
            {
                "rel": "collection",
                "href": flask.url_for(".list_job_results", job_id=job_id, _external=True),  # SHOULD be absolute
                "type": "application/json",
            },
        ],
        "assets": {
            asset_filename: _asset_object(job_id, user_id, asset_filename, asset_metadata, job_info, stac11=False)
        },
        "collection": job_id,
    }
    # Add optional items, if they are present.
    stac_item.update(
        **dict_no_none(
            {
                "epsg": job_info.epsg,
            }
        )
    )
    return stac_item


def _get_job_result_item11(*, batch_jobs: BatchJobs, job_id, item_id, user_id) -> dict:
    if item_id == DriverMlModel.METADATA_FILE_NAME:
        return _download_ml_model_metadata(batch_jobs=batch_jobs, job_id=job_id, file_name=item_id, user_id=user_id)

    metadata = batch_jobs.get_result_metadata(job_id=job_id, user_id=user_id)

    if item_id not in metadata.items:
        raise OpenEOApiException(
            "Item with id {item_id!r} not found in job {job_id!r}".format(item_id=item_id, job_id=job_id),
            status_code=404,
        )
    item_metadata = metadata.items.get(item_id, None)

    job_info = batch_jobs.get_job_info(job_id, user_id)

    assets = {}
    for asset_key, asset in item_metadata.get("assets", {}).items():
        if "output_dir" in asset:
            out_dir = asset.get("output_dir")
            _log.info(f"asset has output dir {out_dir} and href {asset.get('href')}")
            common = os.path.commonpath([asset.get("href"), out_dir])
            href = os.path.relpath(asset.get("href"), common)
        else:
            _log.info(f"asset has no output dir and href {asset.get('href')}")
            href = asset.get("href")
        assets[asset_key] = _asset_object(job_id, user_id, href, asset, job_info, stac11=True)

    properties = item_metadata.get("properties", {"datetime": item_metadata.get("datetime")})
    if properties["datetime"] is None:
        to_datetime = Rfc3339(propagate_none=True).datetime

        start_datetime = item_metadata.get("start_datetime") or to_datetime(job_info.start_datetime)
        end_datetime = item_metadata.get("end_datetime") or to_datetime(job_info.end_datetime)

        if start_datetime == end_datetime:
            properties["datetime"] = start_datetime
        else:
            if start_datetime:
                properties["start_datetime"] = start_datetime
            if end_datetime:
                properties["end_datetime"] = end_datetime

    if job_info.proj_shape:
        properties["proj:shape"] = job_info.proj_shape
    if job_info.proj_bbox:
        properties["proj:bbox"] = job_info.proj_bbox
    if job_info.epsg:
        properties["proj:epsg"] = job_info.epsg
        properties["proj:code"] = f"EPSG:{job_info.epsg}"

    bbox = item_metadata.get("bbox", job_info.bbox)
    if not bbox and job_info.proj_bbox and job_info.epsg:
        bbox = BoundingBox.from_wsen_tuple(job_info.proj_bbox, crs=job_info.epsg).reproject(4326).as_wsen_tuple()
    geometry = item_metadata.get("geometry", job_info.geometry)
    if not geometry and job_info.proj_bbox and job_info.epsg:
        geometry = BoundingBox.from_wsen_tuple(job_info.proj_bbox, crs=job_info.epsg).as_geojson()

    auxiliary_links = [
        _auxiliary_link(link, job_id=job_id, user_id=user_id)
        for link in item_metadata.get("links", [])
        if link.get(ITEM_LINK_PROPERTY.EXPOSE_AUXILIARY, False)
    ]

    stac_item = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            STAC_EXTENSION.EO_V110,
            STAC_EXTENSION.FILEINFO,
            STAC_EXTENSION.PROJECTION_V120,
        ],
        "id": item_id,
        "geometry": geometry,
        "bbox": bbox,
        "properties": properties,
        "links": [
            {
                "rel": "self",
                # MUST be absolute
                "href": flask.url_for(".get_job_result_item11", job_id=job_id, item_id=item_id, _external=True),
                "type": STAC_ITEM_MEDIA_TYPE,
            },
            {
                "rel": "collection",
                "href": flask.url_for(".list_job_results", job_id=job_id, _external=True),  # SHOULD be absolute
                "type": "application/json",
            },
        ]
        + auxiliary_links,
        "assets": assets,
        "collection": job_id,
    }
    # Add optional items, if they are present.
    stac_item.update(
        **dict_no_none(
            {
                "epsg": job_info.epsg,
            }
        )
    )
    return stac_item


def _download_ml_model_metadata(*, batch_jobs: BatchJobs, job_id: str, file_name: str, user_id) -> dict:
    results = batch_jobs.get_result_assets(job_id=job_id, user_id=user_id)
    ml_model_metadata: dict = results.get(file_name, None)
    if ml_model_metadata is None:
        raise FilePathInvalidException(f"{file_name!r} not in {list(results.keys())}")
    assets = deep_get(ml_model_metadata, "assets", default={})
    for asset in assets.values():
        if not asset["href"].startswith("http"):
            asset_file_name = Path(asset["href"]).name
            asset["href"] = get_backend_config().asset_url.build_url(
                asset_metadata=asset, asset_name=asset_file_name, job_id=job_id, user_id=user_id
            )
    stac_item = {
        "stac_version": ml_model_metadata.get("stac_version", "1.0.0"),
        "stac_extensions": ml_model_metadata.get("stac_extensions", []),
        "type": "Feature",
        "id": ml_model_metadata.get("id"),
        "collection": job_id,
        "bbox": ml_model_metadata.get("bbox", []),
        "geometry": ml_model_metadata.get("geometry", {}),
        "properties": ml_model_metadata.get("properties", {}),
        "links": ml_model_metadata.get("links", []),
        "assets": ml_model_metadata.get("assets", {}),
    }
    return stac_item
