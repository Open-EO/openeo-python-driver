"""
Utilities for working with STAC datacube extension

Based on pystac, with adaption layer to compensate for incompatibilities
between available pystac version and actual STAC metadata.
"""
# TODO: move this functionality to the openeo-python-client for better reuse and client-server consistency?

import logging
from typing import Dict, Union
from pathlib import Path

import pystac
from pystac import STACObject
from pystac.extensions.datacube import DatacubeExtension, Dimension, HorizontalSpatialDimension

_log = logging.getLogger(__name__)

StacRef = Union[STACObject, str, Path]


def as_stac_object(stac_ref: StacRef) -> STACObject:
    if isinstance(stac_ref, STACObject):
        return stac_ref
    else:
        return pystac.read_file(stac_ref)


def _get_dimensions(stac_ref: StacRef) -> Dict[str, Dimension]:
    stac_obj: STACObject = as_stac_object(stac_ref)
    # TODO #396 update this to new pystac extension API
    if DatacubeExtension.has_extension(stac_obj):
        cube = DatacubeExtension.ext(stac_obj)
        dimensions = cube.dimensions
    elif any(e.startswith("https://stac-extensions.github.io/datacube/") for e in stac_obj.stac_extensions):
        # TODO #370/#396 as we're currently stuck on an old pystac version that
        #       doesn't support current versions of the datacube extension,
        #       we need workarounds like this
        _log.warning("Forcing pystac datacube extension on possibly unsupported metadata")
        DatacubeExtension.add_to(stac_obj)
        cube = DatacubeExtension.ext(stac_obj)
        dimensions = cube.dimensions
    else:
        dimensions = {}
    return dimensions


def get_spatial_dimensions(stac_ref: StacRef) -> Dict[str, HorizontalSpatialDimension]:
    """
    Get spatial dimensions (HorizontalSpatialDimension) from given STAC object/reference.
    """
    return {
        name: dim
        for name, dim in _get_dimensions(stac_ref=stac_ref).items()
        if isinstance(dim, HorizontalSpatialDimension)
    }
