"""
Utilities to parse STAC metadata
and extract relevant information for OpenEO.

Based on pystac, with adaption layer to compensate for incompatibilities
between available pystac version and actual STAC metadata.
"""
# TODO: move this functionality to the openeo-python-client for better reuse and consistency?

from typing import Dict

import re
import pystac
from pystac import STACObject
import pystac.extensions.datacube
from pystac.extensions.datacube import DatacubeExtension
import logging


_log = logging.getLogger(__name__)


def _get_dimensions(stac_obj: STACObject) -> Dict[str, pystac.extensions.datacube.Dimension]:
    if DatacubeExtension.has_extension(stac_obj):
        cube = DatacubeExtension.ext(stac_obj)
        dimensions = cube.dimensions
    elif any(e.startswith("https://stac-extensions.github.io/datacube/") for e in stac_obj.stac_extensions):
        # TODO #370 because we're currently stuck on old pystac version,
        #       that doesn't support newer versions of the datacube extension,
        #       we have to do tedious DIY workarounds like this
        #       Ideally this should be just a simpl check like
        #           if stac_obj.ext.has("datacube")
        _log.warning("Forcing old datacube extension to work with newer metadata")
        DatacubeExtension.add_to(stac_obj)
        cube = DatacubeExtension.ext(stac_obj)
        dimensions = cube.dimensions
    else:
        dimensions = {}
    return dimensions


def get_spatial_dimensions(stac_obj: STACObject) -> Dict[str, pystac.extensions.datacube.HorizontalSpatialDimension]:
    return {
        k: v
        for k, v in _get_dimensions(stac_obj).items()
        if isinstance(v, pystac.extensions.datacube.HorizontalSpatialDimension)
    }
