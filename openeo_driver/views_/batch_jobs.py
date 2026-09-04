"""

Reusable components and utilities related to the Batch Job view layer.

"""

from typing import List, Union

import flask

from openeo_driver.config import get_backend_config
from openeo_driver.users import user_id_b64_encode
from openeo_driver.views_.utils import add_link_by_rel


def list_job_results_self_link(job_id: str, partial: Union[bool, None] = None) -> dict:
    extra = {}
    if partial:
        # For normalization reasons: only include `partial` when True
        extra["partial"] = "true"
    url = flask.url_for(".list_job_results", job_id=job_id, _external=True, **extra)
    return {
        "rel": "self",
        "href": url,
        # TODO: mime type should probably be "application/geo+json",
        #       but changing that might change too much tests to be feasible for now
        "type": "application/json",
    }


def list_job_results_canonical_link(job_id: str, *, user_id: str, partial: Union[bool, None] = None) -> dict:
    extra = {}
    if partial:
        # For normalization reasons: only include `partial` when True
        extra["partial"] = "true"
    signer = get_backend_config().url_signer
    if signer:
        expires = signer.get_expires()
        secure_key = signer.sign_job_results(job_id=job_id, user_id=user_id, expires=expires)
        user_base64 = user_id_b64_encode(user_id)
        url = flask.url_for(
            ".list_job_results_signed",
            job_id=job_id,
            user_base64=user_base64,
            expires=expires,
            secure_key=secure_key,
            _external=True,
            **extra,
        )
    else:
        url = flask.url_for(".list_job_results", job_id=job_id, _external=True, **extra)

    return {
        "rel": "canonical",
        "href": url,
        # TODO: mime type should probably be "application/geo+json",
        #       but changing that might change too much tests to be feasible for now
        "type": "application/json",
    }


def card4l_link() -> dict:
    return {
        "rel": "card4l-document",
        # TODO: avoid hardcoding this specific URL?
        "href": "http://ceos.org/ard/files/PFS/SR/v5.0/CARD4L_Product_Family_Specification_Surface_Reflectance-v5.0.pdf",
        "type": "application/pdf",
    }


def list_job_results_add_basic_links(
    links: List[dict],
    *,
    job_id: str,
    user_id: str,
    partial: Union[bool, None] = None,
    add_self: bool = True,
    add_canonical: bool = True,
    add_card4l: bool = True,
) -> List[dict]:
    """
    Add basic (self, canonical, ...) links to the given list of links, producing a new list of links.
    """
    if add_self:
        links = add_link_by_rel(links, link=list_job_results_self_link(job_id=job_id, partial=partial), mode="fallback")
    if add_canonical:
        links = add_link_by_rel(
            links,
            link=list_job_results_canonical_link(job_id=job_id, user_id=user_id, partial=partial),
            mode="fallback",
        )
    if add_card4l:
        links = add_link_by_rel(links, link=card4l_link(), mode="fallback")

    return links
