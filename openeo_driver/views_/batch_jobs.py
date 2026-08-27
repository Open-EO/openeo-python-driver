from typing import List, Union, Literal, Iterable

import flask

from openeo_driver.config import get_backend_config
from openeo_driver.users import user_id_b64_encode


def list_job_results_self_link(job_id: str) -> dict:
    params = {k: v for k, v in flask.request.args.items() if k in {"partial"}}
    url = flask.url_for(".list_job_results", job_id=job_id, _external=True, **params)
    return {
        "rel": "self",
        "href": url,
        # TODO: mime type should probably be "application/geo+json",
        #       but changing that might change too much tests to be feasible for now
        "type": "application/json",
    }


def list_job_results_canonical_link(job_id: str, *, user_id: str, partial: Union[bool, None] = None) -> dict:
    extra = {}
    if partial is not None:
        extra["partial"] = str(partial).lower()

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
    Add basic (self, canonical, ...) links in-place to the given links list
    """
    if add_self:
        links = add_link_by_rel(links, link=list_job_results_self_link(job_id=job_id), mode="fallback")
    if add_canonical:
        links = add_link_by_rel(
            links,
            link=list_job_results_canonical_link(job_id=job_id, user_id=user_id, partial=partial),
            mode="fallback",
        )
    if add_card4l:
        links = add_link_by_rel(links, link=card4l_link(), mode="fallback")

    return links


def add_link_by_rel(
    links: Iterable[dict], *, link: dict, mode: Literal["append", "fallback", "overwrite"] = "append"
) -> List[dict]:
    """
    Add a link to the given collection of links, producing a new list links,
    taking care of the "rel" attribute, e.g. to avoid duplicates:
    - "append": always append the new link to the list
    - "fallback": only append the new link if no link with the same rel already exists
    - "overwrite": remove any existing links with the same rel before appending the new link
    """
    # TODO: move this utility to a more generic place

    # Work on a copy
    links = list(links)

    if mode == "append":
        links += [link]
    elif mode == "fallback":
        if not any(l.get("rel") == link.get("rel") for l in links):
            links.append(link)
    elif mode == "overwrite":
        links = [l for l in links if l.get("rel") != link.get("rel")] + [link]
    else:
        raise ValueError(f"Invalid {mode=}")
    return links
