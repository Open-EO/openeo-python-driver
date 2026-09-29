"""
Utilities for working with STAC datacube extension

Based on pystac, with adaption layer to compensate for incompatibilities
between available pystac version and actual STAC metadata.
"""
# TODO: move this functionality to the openeo-python-client for better reuse and client-server consistency?

import logging
from pathlib import Path
from typing import Dict, Union

import openeo.metadata
import pystac
import pystac.extensions.datacube
import pystac.extensions.base


_log = logging.getLogger(__name__)

StacRef = Union[pystac.STACObject, str, Path]


def as_stac_object(stac_ref: StacRef) -> pystac.STACObject:
    if isinstance(stac_ref, pystac.STACObject):
        return stac_ref
    else:
        return pystac.read_file(stac_ref)


def _get_dimensions(stac_ref: StacRef) -> Dict[str, pystac.extensions.datacube.Dimension]:
    stac_obj: pystac.STACObject = as_stac_object(stac_ref)
    if hasattr(stac_obj, "ext") and stac_obj.ext.has("cube"):
        dimensions = stac_obj.ext.cube.dimensions
    else:
        raise ValueError(f"No datacube extension found in STAC object {stac_ref=}")
    return dimensions


def stac_to_cube_metadata(stac_ref: StacRef) -> openeo.metadata.CubeMetadata:
    """
    Parse STAC metadata and convert it to :py:class:`openeo.metadata.CubeMetadata`.
    """
    # Dimensions as pystac objects
    pystac_dimensions = _get_dimensions(stac_ref=stac_ref)

    # Convert to openeo.metadata-style objects
    dimensions = []
    for name, dim in pystac_dimensions.items():
        if isinstance(dim, pystac.extensions.datacube.HorizontalSpatialDimension):
            dimensions.append(
                openeo.metadata.SpatialDimension(
                    name=name,
                    extent=dim.extent,
                    crs=dim.reference_system,
                    step=dim.step,
                )
            )
        elif isinstance(dim, pystac.extensions.datacube.TemporalDimension):
            dimensions.append(
                openeo.metadata.TemporalDimension(
                    name=name,
                    extent=dim.extent,
                )
            )
        elif isinstance(dim, pystac.extensions.datacube.AdditionalDimension) and dim.dim_type == "bands":
            dimensions.append(
                openeo.metadata.BandDimension(
                    name=name,
                    bands=[openeo.metadata.Band(name=b) for b in dim.values],
                )
            )
        else:
            _log.info("Ignoring dimension %s of type %s", name, type(dim))
    return openeo.metadata.CubeMetadata(dimensions=dimensions)
