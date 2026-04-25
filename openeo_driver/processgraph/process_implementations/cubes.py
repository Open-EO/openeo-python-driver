"""Core data cube operation process implementations."""
import logging
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import shapely.geometry
import shapely.ops
from shapely.geometry import GeometryCollection, MultiPolygon

import openeo.udf
from openeo.util import rfc3339

from openeo_driver.datacube import DriverDataCube, DriverVectorCube, SupportsRunUdf
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import DryRunDataCube, DryRunDataTracer
from openeo_driver.errors import (
    FeatureUnsupportedException,
    OpenEOApiException,
    ProcessParameterInvalidException,
)
from openeo_driver.processes import ProcessArgs
from openeo_driver.processgraph.registry import (
    ENV_DRY_RUN_TRACER,
    process,
    process_registry_100,
    process_registry_2xx,
)
from openeo_driver.constants import RESAMPLE_SPATIAL_ALIGNS, RESAMPLE_SPATIAL_METHODS
from openeo_driver.save_result import AggregatePolygonResult, JSONResult, NullResult
from openeo_driver.specs import read_spec
from openeo_driver.util.geometry import geojson_to_geometry, geojson_to_multipolygon
from openeo_driver.utils import EvalEnv

_log = logging.getLogger(__name__)


def _extract_temporal_extent(args: ProcessArgs, field="extent", process_id="filter_temporal") -> Tuple[str, str]:
    extent = args.get_required(name=field)
    if len(extent) != 2:
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="should have length 2, but got {e!r}".format(e=extent)
        )
    start, end = extent[0], extent[1]
    if start is None and end is None:
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="both start and end are null"
        )
    if (start is not None and end is not None
            and rfc3339.parse_date_or_datetime(end) < rfc3339.parse_date_or_datetime(start)):
        raise ProcessParameterInvalidException(
            process=process_id, parameter=field, reason="end '{e}' is before start '{s}'".format(e=end, s=start)
        )
    if start and end and start == end:
        _log.error(
            f"{process_id} {field} with same start and end ({args.get(field)!r}) is invalid and may cause unintended behavior."
            f" This will trigger a TemporalExtentEmpty error in the future."
            f" Instead, use a proper left-closed (end-exclusive) temporal interval."
        )
    return tuple(extent)


def _extract_bbox_extent(args: ProcessArgs, field="extent", process_id="filter_bbox", handle_geojson=False) -> dict:
    extent = args.get_required(name=field)
    if handle_geojson and extent.get("type") in [
        "Polygon", "MultiPolygon", "GeometryCollection", "Feature", "FeatureCollection",
    ]:
        if not _contains_only_polygons(extent):
            raise ProcessParameterInvalidException(
                parameter=field, process=process_id,
                reason="unsupported GeoJSON; requires at least one Polygon or MultiPolygon",
            )
        try:
            w, s, e, n = DriverVectorCube.from_geojson(extent).get_bounding_box()
        except Exception as exc:
            raise ProcessParameterInvalidException(
                parameter=field, process=process_id,
                reason="GeoJSON is not valid: {e!r}".format(e=exc),
            )
        d = {"west": w, "south": s, "east": e, "north": n, "crs": "EPSG:4326"}
    elif all(k in extent for k in ["west", "south", "east", "north"]):
        d = {k: extent[k] for k in ["west", "south", "east", "north"]}
        crs = extent.get("crs") or "EPSG:4326"
        if isinstance(crs, int):
            crs = "EPSG:{crs}".format(crs=crs)
        d["crs"] = crs
    else:
        raise ValueError(f"Failed to extract bounding box from {extent=}")
    return d


def _contains_only_polygons(geojson: dict) -> bool:
    if geojson["type"] in ["Polygon", "MultiPolygon"]:
        return True
    if geojson["type"] == "Feature":
        return _contains_only_polygons(geojson["geometry"])
    if geojson["type"] == "FeatureCollection":
        return all(_contains_only_polygons(feature) for feature in geojson["features"])
    return False


@process
def apply(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    apply_pg = args.get_deep("process", "process_graph", expected_type=dict)
    context = args.get_optional("context", default=None)
    return data_cube.apply(process=apply_pg, context=context, env=env)


@process
def apply_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=(DriverDataCube, DriverVectorCube))
    process_pg = args.get_deep("process", "process_graph", expected_type=dict)
    dimension = args.get_required(
        "dimension", expected_type=str, validator=ProcessArgs.validator_one_of(data_cube.get_dimension_names())
    )
    target_dimension = args.get_optional("target_dimension", default=None, expected_type=str)
    context = args.get_optional("context", default=None)

    cube = data_cube.apply_dimension(
        process=process_pg, dimension=dimension, target_dimension=target_dimension, context=context, env=env
    )
    if target_dimension is not None and target_dimension not in cube.metadata.dimension_names():
        cube = cube.rename_dimension(dimension, target_dimension)
    return cube


@process
def apply_neighborhood(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    process_pg = args.get_deep("process", "process_graph", expected_type=dict)
    size = args.get_required("size")
    overlap = args.get_optional("overlap")
    context = args.get_optional("context", default=None)
    return data_cube.apply_neighborhood(process=process_pg, size=size, overlap=overlap, env=env, context=context)


@process_registry_100.add_function(spec=read_spec("openeo-processes/2.x/proposals/apply_polygon.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/apply_polygon.json"))
def apply_polygon(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    process_pg = args.get_deep("process", "process_graph", expected_type=dict)
    if "polygons" in args and "geometries" not in args:
        _log.warning(
            "DEPRECATED: In process 'apply_polygon': parameter 'polygons' is deprecated, use 'geometries' instead."
        )
        geometries = args.get_required("polygons")
    else:
        geometries = args.get_required("geometries")
    mask_value = args.get_optional("mask_value", expected_type=(int, float), default=None)
    context = args.get_optional("context", default=None)

    if isinstance(geometries, DelayedVector):
        geometries = list(geometries.geometries)
        for p in geometries:
            if not isinstance(p, shapely.geometry.Polygon):
                raise ProcessParameterInvalidException(
                    parameter="polygons", process="apply_polygon",
                    reason="{m!s} is not a polygon.".format(m=p)
                )
        polygon = MultiPolygon(geometries)
    elif isinstance(geometries, DriverVectorCube):
        polygon = geometries.to_multipolygon()
    elif isinstance(geometries, shapely.geometry.base.BaseGeometry):
        polygon = MultiPolygon(geometries)
    elif isinstance(geometries, dict):
        polygon = geojson_to_multipolygon(geometries)
        if isinstance(polygon, shapely.geometry.Polygon):
            polygon = MultiPolygon([polygon])
    else:
        raise ProcessParameterInvalidException(
            parameter="polygons", process="apply_polygon",
            reason=f"unsupported type: {type(geometries).__name__}"
        )

    if polygon.area == 0:
        raise ProcessParameterInvalidException(
            parameter="polygons", process="apply_polygon",
            reason="Polygon {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        )

    return data_cube.apply_polygon(polygons=polygon, process=process_pg, mask_value=mask_value, context=context, env=env)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/chunk_polygon.json"), name="chunk_polygon")
def chunk_polygon(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    reduce_pg = args.get_deep("process", "process_graph", expected_type=dict)
    chunks = args.get_required("chunks")
    mask_value = args.get_optional("mask_value", expected_type=(int, float), default=None)
    context = args.get_optional("context", default=None)

    if isinstance(chunks, DelayedVector):
        polygons = list(chunks.geometries)
        for p in polygons:
            if not isinstance(p, shapely.geometry.Polygon):
                raise ProcessParameterInvalidException(
                    parameter='chunks', process='chunk_polygon',
                    reason="{m!s} is not a polygon.".format(m=p)
                )
        polygon = MultiPolygon(polygons)
    elif isinstance(chunks, shapely.geometry.base.BaseGeometry):
        polygon = MultiPolygon(chunks)
    elif isinstance(chunks, dict):
        polygon = geojson_to_multipolygon(chunks)
        if isinstance(polygon, shapely.geometry.Polygon):
            polygon = MultiPolygon([polygon])
    elif isinstance(chunks, str):
        raise ProcessParameterInvalidException(
            parameter='chunks', process='chunk_polygon',
            reason="Polygon of type string is not yet supported."
        )
    else:
        raise ProcessParameterInvalidException(
            parameter='chunks', process='chunk_polygon',
            reason="Polygon type is not supported."
        )
    if polygon.area == 0:
        raise ProcessParameterInvalidException(
            parameter='chunks', process='chunk_polygon',
            reason="Polygon {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        )

    return data_cube.chunk_polygon(reducer=reduce_pg, chunks=polygon, mask_value=mask_value, context=context, env=env)


@process
def reduce_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    dimension = args.get_required(
        "dimension", expected_type=str, validator=ProcessArgs.validator_one_of(data_cube.get_dimension_names())
    )
    context = args.get_optional("context", default=None)
    return data_cube.reduce_dimension(reducer=reduce_pg, dimension=dimension, context=context, env=env)


@process
def merge_cubes(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube1 = args.get_required("cube1", expected_type=DriverDataCube)
    cube2 = args.get_required("cube2", expected_type=DriverDataCube)
    resolver_process = None
    if "overlap_resolver" in args:
        pg = args.get_deep("overlap_resolver", "process_graph")
        if len(pg) != 1:
            raise ProcessParameterInvalidException(
                parameter='overlap_resolver', process='merge_cubes',
                reason='This backend only supports overlap resolvers with exactly one process for now.')
        resolver_process = next(iter(pg.values()))["process_id"]
    return cube1.merge_cubes(cube2, resolver_process)


@process
def mask(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    mask_cube: DriverDataCube = args.get_required("mask", expected_type=DriverDataCube)
    replacement = args.get_optional("replacement", default=None)
    return cube.mask(mask=mask_cube, replacement=replacement)


@process
def mask_polygon(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube = args.get_required("data", expected_type=DriverDataCube)
    mask_geom = args.get_required("mask")
    replacement = args.get_optional("replacement", default=None)
    inside = args.get_optional("inside", default=False)

    if isinstance(mask_geom, DelayedVector):
        polygon = shapely.ops.unary_union(list(mask_geom.geometries))
    elif isinstance(mask_geom, DriverVectorCube):
        polygon = mask_geom.to_multipolygon()
    elif isinstance(mask_geom, dict) and "type" in mask_geom:
        polygon = geojson_to_multipolygon(mask_geom)
    else:
        raise ProcessParameterInvalidException(
            parameter="mask", process="mask_polygon", reason=f"Unsupported mask type {type(mask_geom)}"
        )

    if polygon.area == 0:
        raise ProcessParameterInvalidException(
            parameter='mask', process='mask_polygon',
            reason="mask {m!s} has an area of {a!r}".format(m=polygon, a=polygon.area)
        )

    return cube.mask_polygon(mask=polygon, replacement=replacement, inside=inside)


@process
def add_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    return data_cube.add_dimension(
        name=args.get_required("name", expected_type=str),
        label=args.get_required("label", expected_type=str),
        type=args.get_optional("type", default="other", expected_type=str),
    )


@process
def drop_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    name: str = args.get_required("name", expected_type=str)
    return cube.drop_dimension(name=name)


@process
def rename_dimension(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    source: str = args.get_required("source", expected_type=str)
    target: str = args.get_required("target", expected_type=str)
    return cube.rename_dimension(source=source, target=target)


@process
def rename_labels(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    dimension: str = args.get_required("dimension", expected_type=str)
    target: List = args.get_required("target", expected_type=list)
    source: Optional[list] = args.get_optional("source", default=None, expected_type=list)
    return cube.rename_labels(dimension=dimension, target=target, source=source)


@process
def dimension_labels(args: ProcessArgs, env: EvalEnv) -> List[str]:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    dimension: str = args.get_required("dimension", expected_type=str)
    return cube.dimension_labels(dimension=dimension)


@process
def filter_temporal(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    start, end = _extract_temporal_extent(args, field="extent", process_id="filter_temporal")
    return cube.filter_temporal(start=start, end=end)


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/filter_labels.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/filter_labels.json"))
def filter_labels(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    condition = args.get_required("condition", expected_type=dict)
    dimension = args.get_required("dimension", expected_type=str)
    context = args.get_optional("context", default=None)
    return cube.filter_labels(condition=condition, dimension=dimension, context=context, env=env)


@process
def filter_bbox(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    spatial_extent = _extract_bbox_extent(args, "extent", process_id="filter_bbox")
    return cube.filter_bbox(**spatial_extent)


@process
def filter_spatial(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    geometries = args.get_required("geometries")

    if isinstance(geometries, dict):
        if "type" in geometries and geometries["type"] != "GeometryCollection":
            geometries = env.backend_implementation.vector_cube_cls.from_geojson(geometries)
        else:
            geometries = geojson_to_geometry(geometries)
            if isinstance(geometries, GeometryCollection):
                polygons = [
                    geom.geoms[0] if isinstance(geom, MultiPolygon) else geom
                    for geom in geometries.geoms
                ]
                geometries = MultiPolygon(polygons)
    elif isinstance(geometries, DelayedVector):
        geometries = DriverVectorCube.from_fiona([geometries.path]).to_multipolygon()
    elif isinstance(geometries, DriverVectorCube):
        pass
    else:
        raise NotImplementedError("filter_spatial does not support {g!r}".format(g=geometries))
    return cube.filter_spatial(geometries)


@process
def filter_bands(args: ProcessArgs, env: EvalEnv) -> Union[DriverDataCube, DriverVectorCube]:
    cube: Union[DriverDataCube, DriverVectorCube] = args.get_required(
        "data", expected_type=(DriverDataCube, DriverVectorCube)
    )
    bands = args.get_required("bands", expected_type=list)
    return cube.filter_bands(bands=bands)


@process
def resample_spatial(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    resolution = args.get_optional(
        "resolution",
        default=0,
        validator=lambda v: isinstance(v, (int, float)) or (isinstance(v, (tuple, list)) and len(v) == 2),
    )
    projection = args.get_optional("projection", default=None)
    method = args.get_enum("method", options=RESAMPLE_SPATIAL_METHODS, default="near")
    align = args.get_enum("align", options=RESAMPLE_SPATIAL_ALIGNS, default="upper-left")
    return cube.resample_spatial(resolution=resolution, projection=projection, method=method, align=align)


@process
def resample_cube_spatial(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    target: DriverDataCube = args.get_required("target", expected_type=DriverDataCube)
    method = args.get_enum("method", options=RESAMPLE_SPATIAL_METHODS, default="near")
    return cube.resample_cube_spatial(target=target, method=method)


@process
def ndvi(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube = args.get_required("data", expected_type=DriverDataCube)
    nir = args.get_optional("nir", default="nir")
    red = args.get_optional("red", default="red")
    target_band = args.get_optional("target_band", default=None)
    return cube.ndvi(nir=nir, red=red, target_band=target_band)


@process
def apply_kernel(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    kernel = np.asarray(args.get_required("kernel", expected_type=list))
    factor = args.get_optional("factor", default=1.0, expected_type=(int, float))
    border = args.get_optional("border", default=0, expected_type=int)
    replace_invalid = args.get_optional("replace_invalid", default=0, expected_type=(int, float))
    return cube.apply_kernel(kernel=kernel, factor=factor, border=border, replace_invalid=replace_invalid)


@process
def linear_scale_range(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    _log.warning("DEPRECATED: linear_scale_range usage directly on cube is deprecated/non-standard.")
    cube: DriverDataCube = args.get_required("x", expected_type=DriverDataCube)
    input_min = args.get_required("inputMin")
    input_max = args.get_required("inputMax")
    output_min = args.get_optional("outputMin", default=0.0)
    output_max = args.get_optional("outputMax", default=1.0)
    return cube.linear_scale_range(input_min, input_max, output_min, output_max)


@process
def aggregate_spatial(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube = args.get_required("data", expected_type=DriverDataCube)
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    target_dimension = args.get_optional("target_dimension", default=None)

    geoms = args.get_required("geometries")

    if isinstance(geoms, DriverVectorCube):
        pass
    elif isinstance(geoms, DryRunDataCube):
        geoms = DriverVectorCube(geometries=gpd.GeoDataFrame(geometry=[]), cube=None)
    elif isinstance(geoms, dict):
        try:
            geoms = env.backend_implementation.vector_cube_cls.from_geojson(geoms)
        except Exception as e:
            _log.error(f"Failed to parse inline GeoJSON geometries in aggregate_spatial: {e!r}", exc_info=True)
            raise ProcessParameterInvalidException(
                parameter="geometries", process="aggregate_spatial",
                reason="Failed to parse inline GeoJSON",
            )
    elif isinstance(geoms, (AggregatePolygonResult, DelayedVector)):
        geoms = geoms.to_driver_vector_cube()
    else:
        raise ProcessParameterInvalidException(
            parameter="geometries", process="aggregate_spatial",
            reason=f"Invalid type: {type(geoms)} ({geoms!r})"
        )
    return cube.aggregate_spatial(geometries=geoms, reducer=reduce_pg, target_dimension=target_dimension)


@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/aggregate_spatial_window.json"))
def aggregate_spatial_window(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube = args.get_required("data", expected_type=DriverDataCube)
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    window_size = args.get_required("size")
    align = args.get_optional("align", default="upper-left")
    pad = args.get_optional("pad", default="pad")
    context = args.get_optional("context", default=None)
    return cube.aggregate_spatial_window(reducer=reduce_pg, size=window_size, align=align, pad=pad, context=context)


@process
def run_udf(args: ProcessArgs, env: EvalEnv):
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    data = args.get_optional(name="data")
    from openeo_driver.processgraph.process_implementations.udp import _get_udf
    udf, runtime = _get_udf(args, env=env)
    context = args.get_optional(name="context", default={})

    if dry_run_tracer and isinstance(data, DryRunDataCube):
        return data.run_udf()

    if runtime.lower() == "EOAP-CWL".lower():
        return env.backend_implementation.run_cwl(data=data, env=env, cwl=udf, context=context)

    if env.get("validation", False):
        raise FeatureUnsupportedException("run_udf is not supported in validation mode.")

    if isinstance(data, SupportsRunUdf) and data.supports_udf(udf=udf, runtime=runtime):
        _log.info(f"run_udf: data of type {type(data)} has direct run_udf support")
        return data.run_udf(udf=udf, runtime=runtime, context=context, env=env)

    if isinstance(data, AggregatePolygonResult):
        pass
    if isinstance(data, DriverVectorCube):
        data = data.to_legacy_save_result()

    if isinstance(data, (DelayedVector, dict)):
        if isinstance(data, dict):
            data = DelayedVector.from_json_dict(data)
        collection = openeo.udf.FeatureCollection(id='VectorCollection', data=data.as_geodataframe())
        data = openeo.udf.UdfData(
            proj={"EPSG": data.crs.to_epsg()}, feature_collection_list=[collection], user_context=context
        )
    elif isinstance(data, JSONResult):
        st = openeo.udf.StructuredData(description="Dictionary data", data=data.get_data(), type="dict")
        data = openeo.udf.UdfData(structured_data_list=[st], user_context=context)
    elif isinstance(data, list):
        data = openeo.udf.UdfData(
            structured_data_list=[openeo.udf.StructuredData(description="Data list", data=data, type="list")],
            user_context=context
        )
    else:
        raise ProcessParameterInvalidException(
            parameter="data", process="run_udf", reason=f"Unsupported data type {type(data)}.",
        )

    from openeo.util import str_truncate
    _log.info(f"[run_udf] Running UDF {str_truncate(udf, width=256)!r} on {data!r}")
    result_data = env.backend_implementation.processing.run_udf(udf, data)
    _log.info(f"[run_udf] UDF resulted in {result_data!r}")

    result_collections = result_data.get_feature_collection_list()
    if result_collections is not None and len(result_collections) > 0:
        geo_data = result_collections[0].data
        dataframe = geo_data
        if isinstance(geo_data, gpd.GeoSeries):
            dataframe = gpd.GeoDataFrame(geometry=geo_data)
        invalid_indexes = dataframe.index[~dataframe.is_valid].tolist()
        if len(invalid_indexes) > 0:
            raise OpenEOApiException(
                status_code=400,
                code="InvalidGeometry",
                message="UDF returned invalid polygons. This could "
                        + f"be due to the input or the code. Invalid index(es): {invalid_indexes}"
            )
        return DriverVectorCube.from_geodataframe(data=dataframe)
    structured_result = result_data.get_structured_data_list()
    if structured_result is not None and len(structured_result) > 0:
        if len(structured_result) == 1:
            return structured_result[0].data
        else:
            return [s.data for s in structured_result]

    raise ProcessParameterInvalidException(
        parameter='udf', process='run_udf',
        reason='The provided UDF should return exactly either a feature collection or a structured result but got: %s .' % str(result_data)
    )
