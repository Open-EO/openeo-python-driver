"""
Utilities for working with STAC datacube extension

Based on pystac, with adaption layer to compensate for incompatibilities
between available pystac version and actual STAC metadata.
"""
# TODO: move this functionality to the openeo-python-client for better reuse and client-server consistency?

import logging
from typing import Dict

from pystac import STACObject
from pystac.extensions.datacube import DatacubeExtension, Dimension, HorizontalSpatialDimension

_log = logging.getLogger(__name__)


def _get_dimensions(stac_obj: STACObject) -> Dict[str, Dimension]:
    if DatacubeExtension.has_extension(stac_obj):
        cube = DatacubeExtension.ext(stac_obj)
        dimensions = cube.dimensions
    elif any(e.startswith("https://stac-extensions.github.io/datacube/") for e in stac_obj.stac_extensions):
        # TODO #370 as we're currently stuck on an old pystac version that
        #       doesn't support newer versions of the datacube extension,
        #       we need workarounds like this
        #       Ideally, this should be just a simple check like `stac_obj.ext.has("datacube")`
        _log.warning("Forcing old datacube extension logic to work with newer metadata")
        DatacubeExtension.add_to(stac_obj)
        cube = DatacubeExtension.ext(stac_obj)
        dimensions = cube.dimensions
    else:
        dimensions = {}
    return dimensions


def get_spatial_dimensions(stac_obj: STACObject) -> Dict[str, HorizontalSpatialDimension]:
    """
    Get spatial dimensions (HorizontalSpatialDimension) from given STAC object.
    """
    return {name: dim for name, dim in _get_dimensions(stac_obj).items() if isinstance(dim, HorizontalSpatialDimension)}
