"""

Implementation of the openEO error handling API
https://open-eo.github.io/openeo-api/errors/

"""
import uuid


class OpenEOApiException(Exception):
    """
    Exception that wraps the fields/data necessary for OpenEO API compliant status/error handling

    see https://open-eo.github.io/openeo-api/errors/#json-error-object:

    required fields:
     - code: standardized textual openEO error code
     - message: human readable explanation

    optional:
     - id: unique identifier for referencing between API responses and server logs
     - url: link to resource that explains error/solutions
    """

    def __init__(self, message, code="Internal", status_code=500, id=None, url=None):
        super().__init__(message)
        # (Standardized) textual openEO error code
        self.code = code
        # HTTP status code
        self.status_code = status_code
        self.id = id or str(uuid.uuid4())
        self.url = url


class CollectionNotFoundException(OpenEOApiException):

    def __init__(self, collection_id):
        super().__init__(
            message="Collection does not exist: {c!r}".format(c=collection_id),
            code="CollectionNotFound",
            status_code=404
        )


class SecondaryServiceNotFound(OpenEOApiException):
    def __init__(self, service_id: str):
        super(SecondaryServiceNotFound, self).__init__(
            message="Service does not exist: {s!r}".format(s=service_id),
            code="ServiceNotFound",
            status_code=404
        )
