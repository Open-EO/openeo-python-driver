"""Data I/O process implementations."""
import logging
import re
from typing import Union

from openeo.metadata import CollectionMetadata

from openeo_driver import dry_run
from openeo_driver.datacube import DriverDataCube, DriverMlModel, DriverVectorCube
from openeo_driver.dry_run import DryRunDataTracer
from openeo_driver.errors import FeatureUnsupportedException, ProcessParameterInvalidException
from openeo_driver.processes import ProcessArgs
from openeo_driver.processgraph.load_params import _extract_load_parameters
from openeo_driver.processgraph.registry import (
    ENV_DRY_RUN_TRACER,
    ENV_SAVE_RESULT,
    process,
    process_registry_100,
    process_registry_2xx,
)
from openeo_driver.save_result import MlModelResult, SaveResult, to_save_result
from openeo_driver.specs import read_spec
from openeo_driver.util.compat import function_has_argument
from openeo_driver.utils import EvalEnv

_log = logging.getLogger(__name__)


def _check_geometry_path_assumption(path: str, process_name: str, parameter: str):
    from openeo.util import str_truncate
    if isinstance(path, str) and path.lstrip().startswith("{"):
        raise ProcessParameterInvalidException(
            parameter=parameter,
            process=process_name,
            reason=f"provided a string (to be handled as path/URL), but it looks like (Geo)JSON encoded data: {str_truncate(path, width=32)!r}.",
        )


def _extract_temporal_extent(args: ProcessArgs, field="extent", process_id="filter_temporal"):
    from openeo.util import rfc3339
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
            f"{process_id} {field} with same start and end ({args.get(field)!r}) is invalid and will trigger a TemporalExtentEmpty error."
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
def load_collection(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    collection_id = args.get_required("id", expected_type=str)

    arguments = {}
    if args.get("temporal_extent"):
        arguments["temporal_extent"] = _extract_temporal_extent(
            args, field="temporal_extent", process_id="load_collection"
        )
    if args.get("spatial_extent"):
        arguments["spatial_extent"] = _extract_bbox_extent(
            args, field="spatial_extent", process_id="load_collection", handle_geojson=True
        )
    if args.contains("bands"):
        arguments["bands"] = args.get_optional(name="bands", expected_type=list)
    if args.contains("properties"):
        arguments["properties"] = args.get_optional(name="properties")
    if args.contains("featureflags"):
        arguments["featureflags"] = args.get_optional(name="featureflags")

    metadata = env.backend_implementation.catalog.get_collection_metadata(collection_id)

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_collection(
            collection_id=collection_id, arguments=arguments, metadata=metadata, env=env, pg_node_id=args.pg_node_id
        )
    else:
        properties = {**CollectionMetadata(metadata).get("_vito", "properties", default={}),
                      **arguments.get("properties", {})}
        source_id = dry_run.DataSource.load_collection(
            collection_id=collection_id,
            properties=properties,
            bands=arguments.get("bands", []),
            env=env,
            pg_node_id=args.pg_node_id,
        ).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)
        load_params.update(arguments)
        kwargs = {}
        if function_has_argument(env.backend_implementation.catalog.load_collection, "pg_node_id"):
            kwargs["pg_node_id"] = args.pg_node_id
        return env.backend_implementation.catalog.load_collection(
            collection_id=collection_id, load_params=load_params, env=env, **kwargs
        )


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/query_stac.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/query_stac.json"))
def query_stac(args: ProcessArgs, env: EvalEnv) -> dict:
    url = args.get_required("url")
    temporal_extent = None
    spatial_extent = None
    if args.get("temporal_extent"):
        temporal_extent = _extract_temporal_extent(args, field="temporal_extent", process_id="query_stac")
    if args.get("spatial_extent"):
        spatial_extent = _extract_bbox_extent(args, field="spatial_extent", process_id="query_stac")

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        _log.warning("Dry run tracer not supported for query_stac")
        return {}
    else:
        return env.backend_implementation.query_stac(
            url=url, spatial_extent=spatial_extent, temporal_extent=temporal_extent, env=env
        )


@process
def save_result(args: ProcessArgs, env: EvalEnv) -> SaveResult:
    data = args.get_required("data")
    format = args.get_required("format", expected_type=str)
    options = args.get_optional("options", expected_type=dict, default={})

    if isinstance(data, SaveResult):
        data = data.with_format(format, options)
        if ENV_SAVE_RESULT in env:
            env[ENV_SAVE_RESULT].append(data)
        return data
    else:
        result = to_save_result(data, format=format, options=options)
        if ENV_SAVE_RESULT in env:
            env[ENV_SAVE_RESULT].append(result)
            return data
        else:
            return result


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/save_ml_model.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/save_ml_model.json"))
def save_ml_model(args: ProcessArgs, env: EvalEnv) -> MlModelResult:
    data = args.get_required("data", expected_type=DriverMlModel)
    options = args.get_optional("options", default={}, expected_type=dict)
    return MlModelResult(ml_model=data, options=options)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/load_ml_model.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/load_ml_model.json"))
def load_ml_model(args: ProcessArgs, env: EvalEnv) -> DriverMlModel:
    if env.get(ENV_DRY_RUN_TRACER):
        return DriverMlModel()
    job_id = args.get_required("id", expected_type=str)
    return env.backend_implementation.load_ml_model(job_id)


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/load_uploaded_files.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_uploaded_files.json"))
def load_uploaded_files(args: ProcessArgs, env: EvalEnv) -> Union[DriverVectorCube, DriverDataCube]:
    paths = args.get_required("paths", expected_type=list)
    format = args.get_required(
        "format",
        expected_type=str,
        validator=ProcessArgs.validator_file_format(formats=env.backend_implementation.file_formats()["input"]),
    )
    options = args.get_optional("options", default={})

    if DriverVectorCube.from_fiona_supports(format):
        return DriverVectorCube.from_fiona(paths, driver=format, options=options)
    else:
        dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
        if dry_run_tracer:
            return dry_run_tracer.load_uploaded_files(paths=paths, format=format, options=options)
        else:
            source_id = dry_run.DataSource.load_uploaded_files(
                paths=paths, format=format, options=options
            ).get_source_id()
            load_params = _extract_load_parameters(env, source_id=source_id)
            return env.backend_implementation.load_uploaded_files(
                paths=paths, format=format, options=options, load_params=load_params, env=env
            )


@process_registry_100.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_geojson.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_geojson.json"))
def load_geojson(args: ProcessArgs, env: EvalEnv) -> DriverVectorCube:
    data = args.get_required(
        "data",
        validator=ProcessArgs.validator_geojson_dict(
            allowed_types=["Point", "MultiPoint", "Polygon", "MultiPolygon", "Feature", "FeatureCollection"]
        ),
    )
    properties = args.get_optional("properties", default=[], expected_type=(list, tuple))
    vector_cube = env.backend_implementation.vector_cube_cls.from_geojson(data, columns_for_cube=properties)
    return vector_cube


@process_registry_100.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_url.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_url.json"))
def load_url(args: ProcessArgs, env: EvalEnv) -> DriverVectorCube:
    url = args.get_required("url", expected_type=str, validator=re.compile("^(https?|s3)://").match)
    format = args.get_required(
        "format",
        expected_type=str,
        validator=ProcessArgs.validator_file_format(formats=env.backend_implementation.file_formats()["input"]),
    )
    options = args.get_optional("options", default={})

    if DriverVectorCube.from_fiona_supports(format):
        return DriverVectorCube.from_fiona(paths=[url], driver=format, options=options)
    else:
        raise FeatureUnsupportedException(f"Loading format {format!r} is not supported")


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/load_result.json"))
def load_result(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    _log.warning("DEPRECATED: load_result usage")
    job_id = args.get_required("id", expected_type=str)
    user = env.get("user")

    arguments = {}
    if args.contains("temporal_extent"):
        arguments["temporal_extent"] = _extract_temporal_extent(
            args, field="temporal_extent", process_id="load_result"
        )
    if args.contains("spatial_extent"):
        arguments["spatial_extent"] = _extract_bbox_extent(
            args, field="spatial_extent", process_id="load_result", handle_geojson=True
        )
    if args.contains("bands"):
        arguments["bands"] = args.get_optional(name="bands", expected_type=list)

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_result(job_id, arguments)
    else:
        source_id = dry_run.DataSource.load_result(job_id).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)
        return env.backend_implementation.load_result(
            job_id=job_id, user_id=user.user_id if user is not None else None,
            load_params=load_params, env=env
        )


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/load_stac.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/load_stac.json"))
def load_stac(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    url: str = args.get_required(name="url", expected_type=str)

    arguments = {}
    if args.get("temporal_extent"):
        arguments["temporal_extent"] = _extract_temporal_extent(
            args, field="temporal_extent", process_id="load_stac"
        )
    if args.get("spatial_extent"):
        arguments["spatial_extent"] = _extract_bbox_extent(
            args, field="spatial_extent", process_id="load_stac", handle_geojson=True
        )
    if args.contains("bands"):
        arguments["bands"] = args.get_optional(name="bands", expected_type=list)
    if args.contains("properties"):
        arguments["properties"] = args.get_optional(name="properties")
    if args.contains("featureflags"):
        arguments["featureflags"] = args.get_optional(name="featureflags")

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        return dry_run_tracer.load_stac(url=url, arguments=arguments, env=env, pg_node_id=args.pg_node_id)
    else:
        source_id = dry_run.DataSource.load_stac(
            url,
            properties=arguments.get("properties", {}),
            bands=arguments.get("bands", []),
            env=env,
            pg_node_id=args.pg_node_id,
        ).get_source_id()
        load_params = _extract_load_parameters(env, source_id=source_id)
        load_params.resolve_tile_overlap = False
        load_params.update(arguments)

        kwargs = {}
        if function_has_argument(env.backend_implementation.load_stac, "pg_node_id"):
            kwargs["pg_node_id"] = args.pg_node_id

        return env.backend_implementation.load_stac(url=url, load_params=load_params, env=env, **kwargs)


@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/export_workspace.json"))
def export_workspace(args: ProcessArgs, env: EvalEnv) -> SaveResult:
    data = args.get_required("data")
    workspace_id = args.get_required("workspace", expected_type=str)
    merge = args.get_optional("merge", expected_type=str)

    if isinstance(data, SaveResult):
        result = data
    else:
        results = env[ENV_SAVE_RESULT]
        result = results[-1]

    result.add_workspace_export(workspace_id, merge=merge)
    return result
