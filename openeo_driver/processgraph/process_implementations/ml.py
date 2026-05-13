"""Machine learning process implementations."""
import logging
from typing import Union

import shapely.geometry

from openeo_driver.datacube import DriverDataCube, DriverMlModel, DriverVectorCube
from openeo_driver.dry_run import DryRunDataTracer
from openeo_driver.errors import ProcessParameterInvalidException
from openeo_driver.processes import ProcessArgs
from openeo_driver.processgraph.registry import (
    ENV_DRY_RUN_TRACER,
    NoPythonImplementationError,
    process_registry_100,
    process_registry_2xx,
)
from openeo_driver.save_result import AggregatePolygonSpatialResult
from openeo_driver.specs import read_spec
from openeo_driver.utils import EvalEnv

_log = logging.getLogger(__name__)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/fit_class_random_forest.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/fit_class_random_forest.json"))
def fit_class_random_forest(args: ProcessArgs, env: EvalEnv) -> DriverMlModel:
    if env.get(ENV_DRY_RUN_TRACER):
        return DriverMlModel()

    predictors = args.get_required(
        name="predictors",
        expected_type=(AggregatePolygonSpatialResult, DriverVectorCube),
        expected_type_name="non-temporal vector cube",
    )

    target = args.get_required(
        name="target", expected_type=(dict, DriverVectorCube), expected_type_name="feature collection or vector cube"
    )
    if isinstance(target, DriverVectorCube):
        target: dict = shapely.geometry.mapping(target.get_geometries())
    if not (isinstance(target, dict) and target.get("type") == "FeatureCollection"):
        raise ProcessParameterInvalidException(
            parameter="target",
            process=args.process_id,
            reason=f"expected feature collection or vector-cube value, but got {type(target)}.",
        )

    num_trees = args.get_optional(name="num_trees", expected_type=int, default=100)
    if not isinstance(num_trees, int) or num_trees < 0:
        raise ProcessParameterInvalidException(
            parameter="num_trees", process=args.process_id, reason="should be an integer larger than 0."
        )
    max_variables = args.get_optional(name="max_variables", expected_type=(int, str))
    mtry = args.get_optional(name="max_variables", expected_type=(int, str))
    if max_variables is None and mtry:
        _log.warning(f"{args.process_id}: usage of deprecated argument 'mtry'. Use 'max_variables' instead.")
        max_variables = mtry

    seed = args.get_optional(name="seed", expected_type=int)

    return predictors.fit_class_random_forest(
        target=target, num_trees=num_trees, max_variables=max_variables, seed=seed,
    )


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/fit_class_catboost.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/fit_class_catboost.json"))
def fit_class_catboost(args: ProcessArgs, env: EvalEnv) -> DriverMlModel:
    if env.get(ENV_DRY_RUN_TRACER):
        return DriverMlModel()

    predictors = args.get_required("predictors", expected_type=(AggregatePolygonSpatialResult, DriverVectorCube))

    target: Union[dict, DriverVectorCube] = args.get_required("target")
    if isinstance(target, DriverVectorCube):
        target: dict = shapely.geometry.mapping(target.get_geometries())
    if not (isinstance(target, dict) and target.get("type") == "FeatureCollection"):
        raise ProcessParameterInvalidException(
            parameter="target",
            process=args.process_id,
            reason=f"expected feature collection or vector-cube value, but got {type(target)}.",
        )

    def get_validated_parameter(args, param_name, default_value, expected_type, min_value=1, max_value=1000):
        return args.get_optional(
            param_name,
            default=default_value,
            expected_type=expected_type,
            validator=ProcessArgs.validator_generic(
                lambda v: v >= min_value and v <= max_value,
                error_message=f"The `{param_name}` parameter should be an integer between {min_value} and {max_value}.",
            ),
        )

    iterations = get_validated_parameter(args, "iterations", 5, int, 1, 500)
    depth = get_validated_parameter(args, "depth", 5, int, 1, 16)
    seed = get_validated_parameter(args, "seed", 0, int, 0, 2 ** 31 - 1)

    return predictors.fit_class_catboost(target=target, iterations=iterations, depth=depth, seed=seed)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/predict_onnx.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/predict_onnx.json"))
def predict_onnx(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    model = args.get_required("model", expected_type=str)
    return data_cube.predict_onnx(model=model)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/predict_random_forest.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/predict_random_forest.json"))
def predict_random_forest(args: ProcessArgs, env: EvalEnv):
    raise NoPythonImplementationError


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/predict_catboost.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/predict_catboost.json"))
def predict_catboost(args: ProcessArgs, env: EvalEnv):
    raise NoPythonImplementationError


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/predict_probabilities.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/predict_probabilities.json"))
def predict_probabilities(args: ProcessArgs, env: EvalEnv):
    raise NoPythonImplementationError
