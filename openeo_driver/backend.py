"""

Base structure of a backend implementation (e.g. Geotrellis based)
to be exposed with HTTP REST frontend.

It is organised in microservice-like parts following https://open-eo.github.io/openeo-api/apireference/
to allow composability, isolation and better reuse.

Also see https://github.com/Open-EO/openeo-python-driver/issues/8
"""

from typing import List, Tuple

from openeo_driver.errors import OpenEOApiException


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


class OpenEoBackendImplementation:
    """
    Simple container of all openEo "microservices"
    """

    def __init__(self, secondary_services: SecondaryServices):
        self.secondary_services = secondary_services
