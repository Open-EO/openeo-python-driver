# TODO: rename this module to something in snake case? It doesn't even implement a ProcessGraphDeserializer class.
# TODO: and related: separate generic process graph handling from more concrete openEO process implementations
#
# NOTE: This file is now a backward-compatibility shim.
#       All implementations have been moved to openeo_driver/processgraph/.
#       Import from the new location for new code.

# pylint: disable=unused-import

from openeo_driver.processgraph.registry import (
    # Registry objects
    process_registry_100,
    process_registry_2xx,
    # Decorators
    process,
    simple_function,
    non_standard_process,
    custom_process,
    custom_process_from_process_graph,
    # Helper functions
    _add_standard_processes,
    _process_function_from_process_graph,
    _register_fallback_implementations_by_process_graph,
    # Internal lists
    _OPENEO_PROCESSES_PYTHON_WHITELIST,
    _openeo_processes_extra,
    # Type alias
    ProcessFunction,
    # Constants
    ENV_SOURCE_CONSTRAINTS,
    ENV_DRY_RUN_TRACER,
    ENV_FINAL_RESULT,
    ENV_SAVE_RESULT,
    ENV_MAX_BUFFER,
    # Classes
    NoPythonImplementationError,
    SimpleProcessing,
    ConcreteProcessing,
)

from openeo_driver.processgraph.evaluator import (
    DEFAULT_TEMPORAL_EXTENT,
    _collect_end_nodes,
    _end_node_ids,
    evaluate,
    convert_node,
    apply_process,
    _evaluate_process_graph_process,
    evaluate_udp,
    evaluate_process_from_url,
    flatten_children_node_types,
    flatten_children_node_names,
    check_subgraph_for_data_mask_optimization,
    collect,
)

from openeo_driver.processgraph.load_params import (
    _collection_crs,
    _collection_resolution,
    _align_extent,
    _extract_load_parameters,
)

# Process implementations (imported for side effects: registry decoration)
from openeo_driver.processgraph.process_implementations.io import (
    load_collection,
    query_stac,
    save_result,
    save_ml_model,
    load_ml_model,
    load_uploaded_files,
    load_geojson,
    load_url,
    load_result,
    load_stac,
    export_workspace,
    _check_geometry_path_assumption,
    _extract_temporal_extent,
    _extract_bbox_extent,
    _contains_only_polygons,
)

from openeo_driver.processgraph.process_implementations.cubes import (
    apply,
    apply_dimension,
    apply_neighborhood,
    apply_polygon,
    chunk_polygon,
    reduce_dimension,
    merge_cubes,
    mask,
    mask_polygon,
    add_dimension,
    drop_dimension,
    rename_dimension,
    rename_labels,
    dimension_labels,
    filter_temporal,
    filter_labels,
    filter_bbox,
    filter_spatial,
    filter_bands,
    resample_spatial,
    resample_cube_spatial,
    ndvi,
    apply_kernel,
    linear_scale_range,
    aggregate_spatial,
    aggregate_spatial_window,
    run_udf,
)

from openeo_driver.processgraph.process_implementations.temporal import (
    aggregate_temporal,
    aggregate_temporal_period,
    _period_to_intervals,
)

from openeo_driver.processgraph.process_implementations.ml import (
    fit_class_random_forest,
    fit_class_catboost,
    predict_onnx,
    predict_random_forest,
    predict_catboost,
    predict_probabilities,
)

from openeo_driver.processgraph.process_implementations.geometry import (
    vector_buffer,
    get_geometries,
    read_vector,
    to_vector_cube,
    raster_to_vector,
    vector_to_raster,
)

from openeo_driver.processgraph.process_implementations.sar import (
    atmospheric_correction,
    sar_backscatter,
    resolution_merge,
    mask_scl_dilation,
    to_scl_dilation_mask,
    mask_l1c,
)

from openeo_driver.processgraph.process_implementations.misc import (
    constant,
    inspect,
    sleep,
    discard_result,
    date_shift,
    date_between,
    if_,
)

from openeo_driver.processgraph.process_implementations.array import (
    array_append,
    array_interpolate_linear,
    array_concat,
    array_create,
    array_apply,
)

from openeo_driver.processgraph.process_implementations.terrain import (
    aspect,
    slope,
    convert_data_type,
)

from openeo_driver.processgraph.process_implementations.text import (
    text_begins,
    text_contains,
    text_ends,
    text_merge,
    text_concat,
)
