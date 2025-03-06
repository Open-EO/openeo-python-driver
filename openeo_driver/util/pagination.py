import flask
from typing import Union, Optional

import dataclasses


@dataclasses.dataclass(frozen=True)
class PaginationRequest:
    """
    Representation of pagination related request parameters.

    The openEO API just defines a request parameter `limit`
    as a trigger to enable pagination.
    Additional request parameters or URL patterns
    to encode page numbers/offsets are not defined in the API,
    and open for backends to choose.
    """

    limit: Union[None, int]
    request_parameters: dict

    @classmethod
    def from_request(cls, request: flask.Request):
        limit = request.args.get("limit", default=None, type=int)
        request_parameters = flask.request.args.to_dict()
        return cls(limit=limit, request_parameters=request_parameters)

    def __bool__(self):
        return self.limit is not None
