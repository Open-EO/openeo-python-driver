"""

Base structure of a backend implementation (e.g. Geotrellis based)
to be exposed with HTTP REST frontend.

It is organised in microservice-like parts following https://open-eo.github.io/openeo-api/apireference/
to allow composability, isolation and better reuse.

Also see https://github.com/Open-EO/openeo-python-driver/issues/8
"""

import copy
import warnings
from pathlib import Path
from typing import List, Tuple, Union, Dict

from openeo import ImageCollection
from openeo_driver.errors import OpenEOApiException, CollectionNotFoundException
from openeo_driver.utils import read_json


class SecondaryServices:
    """
    Base contract/implementation for Secondary Services "microservice"
    https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management
    """

    def service_types(self) -> dict:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1service_types/get"""
        return {}

    def list_services(self) -> List[dict]:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services/get"""
        # TODO require auth/user handle?
        return []

    def service_info(self, service_id: str) -> dict:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services~1{service_id}/get"""
        # TODO require auth/user handle?
        raise NotImplementedError()

    def create_service(self, data: dict) -> Tuple[str, str]:
        """
        https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services/post
        :return: (location, openeo_identifier)
        """
        from openeo_driver.ProcessGraphDeserializer import evaluate
        # TODO require auth/user handle?
        process_graph = data['process_graph']

        service_type = data['type']
        if service_type.lower() not in set(st.lower() for st in self.service_types()):
            raise OpenEOApiException(
                message="Secondary service type {t!r} is not supported.".format(t=service_type),
                code="ServiceUnsupported", status_code=400
            )

        # TODO: avoid passing api version?
        api_version = data.pop('api_version')
        image_collection = evaluate(process_graph, viewingParameters={'version': api_version})
        service_info = image_collection.tiled_viewing_service(**data)
        return service_info['url'], service_info.get('service_id', 'unknown')

    def update_service(self, service_id: str, data: dict) -> None:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services~1{service_id}/patch"""
        # TODO require auth/user handle?
        raise NotImplementedError()

    def remove_service(self, service_id: str) -> None:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1services~1{service_id}/delete"""
        # TODO require auth/user handle?
        raise NotImplementedError()

    # TODO https://open-eo.github.io/openeo-api/apireference/#tag/Secondary-Services-Management/paths/~1subscription/get


class CollectionCatalog:
    """
    Basic implementation of a catalog of collections/EO data
    """

    _stac_version = "0.7.0"

    def __init__(self, all_metadata: List[dict]):
        self._catalog = {layer["id"]: dict(layer) for layer in all_metadata}

    @classmethod
    def from_json_file(cls, filename: Union[str, Path] = "layercatalog.json", *args, **kwargs):
        """Factory to read catalog from a JSON file"""
        return cls(read_json(filename), *args, **kwargs)

    def assert_collection_id(self, collection_id):
        if collection_id not in self._catalog:
            raise CollectionNotFoundException(collection_id)

    def get_all_metadata(self) -> List[dict]:
        """
        Basic metadata for all datasets
        https://open-eo.github.io/openeo-api/apireference/#tag/EO-Data-Discovery/paths/~1collections/get
        :return:
        """
        return [self._normalize_layer_metadata(m) for m in self._catalog.values()]

    def get_collection_metadata(self, collection_id) -> dict:
        """
        Full metadata for a specific dataset
        https://open-eo.github.io/openeo-api/apireference/#tag/EO-Data-Discovery/paths/~1collections~1{collection_id}/get
        :param collection_id:
        :return:
        """

        self.assert_collection_id(collection_id)
        return self._normalize_layer_metadata(self._catalog[collection_id])

    def load_collection(self, collection_id: str, viewing_parameters: dict) -> ImageCollection:
        raise NotImplementedError

    def _normalize_layer_metadata(self, metadata: dict) -> dict:
        """
        Make sure the given layer metadata roughly complies to OpenEO spec.
        """
        metadata = copy.deepcopy(metadata)

        # Metadata should at least contain an id.
        collection_id = metadata["id"]

        # Make sure required fields are set.
        metadata.setdefault("stac_version", self._stac_version)
        metadata.setdefault("links", [])
        metadata.setdefault("other_properties", {})
        # Warn about missing fields where sensible defaults are not feasible
        fallbacks = {
            "description": "Description of {c} (#TODO)".format(c=collection_id),
            "license": "proprietary",
            "extent": {"spatial": [0, 0, 0, 0], "temporal": [None, None]},
            "properties": {"cube:dimensions": {}},
        }
        for key, value in fallbacks.items():
            if key not in metadata:
                warnings.warn(
                    "Collection {c} is missing required metadata field {k!r}.".format(c=collection_id, k=key),
                    category=CollectionIncompleteMetadataWarning
                )
                metadata[key] = value

        return metadata


class CollectionIncompleteMetadataWarning(UserWarning):
    pass


class OpenEoBackendImplementation:
    """
    Simple container of all openEo "microservices"
    """

    def __init__(self, secondary_services: SecondaryServices, catalog: CollectionCatalog):
        self.secondary_services = secondary_services
        self.catalog = catalog

    def health_check(self) -> str:
        return "OK"

    def output_formats(self) -> dict:
        """https://open-eo.github.io/openeo-api/apireference/#tag/Capabilities/paths/~1output_formats/get"""
        return {}

    def load_disk_data(self, format: str, glob_pattern: str, options: dict, viewing_parameters: dict) -> object:
        raise NotImplementedError
