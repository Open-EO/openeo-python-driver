
# Changelog

All notable changes to this project will be documented in this file.

This project does not have a real release cycle (yet).
Upstream projects usually depend on development snapshots of this project.
Still, to have some kind of indicator of small versus big change,
we try to bump the version number (in `openeo_driver/_version.py`)
roughly according to [Semantic Versioning](https://semver.org/).

When adding a feature/bugfix without bumping the version number:
just describe it under the "In progress" section.
When bumping the version number in `openeo_driver/_version.py`
(possibly accompanying a feature/bugfix):
"close" the "In Progress" section by changing its title to the new version number
(and describe accompanying changes, if any, under it too)
and start a new "In Progress" section above it.

<!-- start-of-changelog -->

## In progress

# 0.113.0

- Add `max_updated_ago` to `JobRegistryInterface.list_active_jobs` API  ([Open-EO/openeo-geopyspark-driver#902](https://github.com/Open-EO/openeo-geopyspark-driver/issues/902))

# 0.112.0
- Support exporting objects to object storage workspace ([eu-cdse/openeo-cdse-infra#278](https://github.com/eu-cdse/openeo-cdse-infra/issues/278))
- Move ObjectStorageWorkspace implementation to openeo-geopyspark-driver ([eu-cdse/openeo-cdse-infra#278](https://github.com/eu-cdse/openeo-cdse-infra/issues/278))

# 0.111.1

- Remove `JobRegistryInterface.list_trackable_jobs` API ([Open-EO/openeo-geopyspark-driver#902](https://github.com/Open-EO/openeo-geopyspark-driver/issues/902))

# 0.111.0

- Add `has_application_id` argument to `JobRegistryInterface.list_active_jobs` in preparation to eliminate `list_trackable_jobs` ([Open-EO/openeo-geopyspark-driver#902](https://github.com/Open-EO/openeo-geopyspark-driver/issues/902))

# 0.110.0

- Add `max_age` support to `ElasticJobRegistry.list_trackable_jobs` ([Open-EO/openeo-geopyspark-driver#902](https://github.com/Open-EO/openeo-geopyspark-driver/issues/902))

# 0.109.0

- Support multiple `export_workspace` processes ([eu-cdse/openeo-cdse-infra#264](https://github.com/eu-cdse/openeo-cdse-infra/issues/264))
- Fix `export_workspace` process not executed in process graph with multiple `save_result` processes ([eu-cdse/openeo-cdse-infra#264](https://github.com/eu-cdse/openeo-cdse-infra/issues/264))
- Restore deterministic evaluation of process graph with multiple end nodes

# 0.108.0

- Added support for `apply_vectorcube` UDF signature in `run_udf_code` ([Open-EO/openeo-geopyspark-driver#881]https://github.com/Open-EO/openeo-geopyspark-driver/issues/881)

## 0.107.8

- add `check_config_definition` helper to check definition of `OpenEoBackendConfig` based configs

## 0.107.7

- return STAC Items with valid date/time for time series job results ([Open-EO/openeo-geopyspark-driver#852](https://github.com/Open-EO/openeo-geopyspark-driver/issues/852))

## 0.107.6

- support passing the output of `raster_to_vector` to `aggregate_spatial` during dry run ([EU-GRASSLAND-WATCH/EUGW#7](https://github.com/EU-GRASSLAND-WATCH/EUGW/issues/7))
- support `vector_to_raster` of geometries not in EPSG:4326 ([EU-GRASSLAND-WATCH/EUGW#7](https://github.com/EU-GRASSLAND-WATCH/EUGW/issues/7))

## 0.107.5

- Return compliant GeoJSON from `DriverVectorCube#get_bounding_box_geojson` ([Open-EO/openeo-geopyspark-driver#854](https://github.com/Open-EO/openeo-geopyspark-driver/issues/854))

## 0.107.4

- Don't require a `final_result` entry in the `EvalEnv` in `convert_node` ([openeo-aggregator#151](https://github.com/Open-EO/openeo-aggregator/issues/151))

## 0.107.3

- Support `save_result` processes in arbitrary subtrees in the process graph i.e. those not necessarily contributing to the final result ([Open-EO/openeo-geopyspark-driver#424](https://github.com/Open-EO/openeo-geopyspark-driver/issues/424))

## 0.107.2

- Fix default level of `inspect` process (defaults to `info`) ([Open-EO/openeo-geopyspark-driver#424](https://github.com/Open-EO/openeo-geopyspark-driver/issues/424))
- `apply_polygon`: add support for `geometries` argument (in addition to legacy, but still supported `polygons`) ([Open-EO/openeo-processes#511](https://github.com/Open-EO/openeo-processes/issues/511))

## 0.107.1

- Update to "remote-process-definition" extension (originally called "remote-udp")
  ([#297](https://github.com/Open-EO/openeo-python-driver/issues/297), [Open-EO/openeo-api#540](https://github.com/Open-EO/openeo-api/issues/540))

## 0.107.0

- `evaluate_process_from_url`: drop support for URL guessing from folder-like URL ([#297)](https://github.com/Open-EO/openeo-python-driver/issues/297))
- `evaluate_process_from_url`: align with new (and experimental) "remote-udp" extension ([#297)](https://github.com/Open-EO/openeo-python-driver/issues/297))

## 0.106.0

- Add API to define conformance classes to `OpenEoBackendImplementation`

## 0.105.0

- Require at least `werkzeug>=3.0.3` ([#281](https://github.com/Open-EO/openeo-python-driver/issues/281))

## 0.104.0

- Expose CSV/GeoParquet output assets as STAC items ([Open-EO/openeo-geopyspark-driver#787](https://github.com/Open-EO/openeo-geopyspark-driver/issues/787))

## 0.103.2

- Start warning about deprecated `evaluate_process_from_url` usage (eu-cdse/openeo-cdse-infra#167)

## 0.103.0, 0.103.1

- Add helper for finding changelog path

## 0.102.2

- Support `DriverVectorCube` in `apply_polygon` ([#287](https://github.com/Open-EO/openeo-python-driver/issues/287))

## 0.102.0

- Emit "in" operator ([Open-EO/openeo-opensearch-client#32](https://github.com/Open-EO/openeo-opensearch-client/issues/32),
  [Open-EO/openeo-geopyspark-driver/#776](https://github.com/Open-EO/openeo-geopyspark-driver/issues/776))

## 0.101.0

- Add simple enum `AUTHENTICATION_METHOD` for `User.internal_auth_data.get("authentication_method")` values

## 0.100.0

- Rename `BatchJobLoggingFilter` to more general applicable `GlobalExtraLoggingFilter`

## 0.99.0

- Support `job_options` in synchronous processing (experimental)
  (related to [Open-EO/openeo-geopyspark-driver#531](https://github.com/Open-EO/openeo-geopyspark-driver/issues/531), eu-cdse/openeo-cdse-infra#114)

## 0.98.0

- Add `job_options` argument to `OpenEoBackendImplementation.request_costs()` API.
  It's optional and unused for now, but allows openeo-geopyspark-driver to adapt already.
  (related to [Open-EO/openeo-geopyspark-driver#531](https://github.com/Open-EO/openeo-geopyspark-driver/issues/531), eu-cdse/openeo-cdse-infra#114)

## 0.97.0

- Remove deprecated and now unused `user_id` argument from `OpenEoBackendImplementation.request_costs()`
  (cleanup related to [Open-EO/openeo-geopyspark-driver#531](https://github.com/Open-EO/openeo-geopyspark-driver/issues/531))

## 0.96.2

- Decreased default ttl in `ClientCredentialsAccessTokenHelper` to 5 minutes

## 0.96.1

- Fix delete in EJR CLI app

## 0.96.0

- Add rudimentary multi-project changelog support

## 0.95.2

- Automatically add job_id and user_id to all logs during job start handling
  ([#214](https://github.com/Open-EO/openeo-python-driver/issues/214), eu-cdse/openeo-cdse-infra#56)

## 0.95.1

- Enable `ExtraLoggingFilter` by default from `get_logging_config`
  ([#214](https://github.com/Open-EO/openeo-python-driver/issues/214))

## 0.95.0

- Add `ExtraLoggingFilter` for context manager based "extra" logging
  ([#214](https://github.com/Open-EO/openeo-python-driver/issues/214), [#233](https://github.com/Open-EO/openeo-python-driver/pull/233))

## 0.94.2

- Fix dry run flow for aggregate_spatial, run_udf, and vector_to_raster ([#276](https://github.com/Open-EO/openeo-python-driver/issues/276)).

## 0.94.1

- Improve resilience by retrying EJR search requests ([Open-EO/openeo-geopyspark-driver#720](https://github.com/Open-EO/openeo-geopyspark-driver/issues/720)).

## 0.93.0

- For client credentials: use OIDC "sub" identifier as user_id instead of config based mapping to be compatible
  with ETL API reporting requirements ([Open-EO/openeo-geopyspark-driver#708](https://github.com/Open-EO/openeo-geopyspark-driver/issues/708))

## 0.92.0

- Reinstate the `werkzeug<3` constraint. Apparently too many deployments are stuck with a very low Flask version,
  which is not compatible with Werkzeug 3 ([#243](https://github.com/Open-EO/openeo-python-driver/issues/243)).
  Pinning this down in openeo-python-driver is unfortunately the most feasible solution for now.

## 0.91.0

- Support `export_workspace` process and `DiskWorkspace` implementation ([Open-EO/openeo-geopyspark-driver#676](https://github.com/Open-EO/openeo-geopyspark-driver/issues/676))


## 0.90.1

- Fix picking up `flask_settings` from OpenEoBackendConfig.
  This introduces/enables a default maximum request size (`MAX_CONTENT_LENGTH`) of 2MB  ([#254](https://github.com/Open-EO/openeo-python-driver/issues/254))

## 0.90.0

- Drop werkzeug<3 constraint ([#243](https://github.com/Open-EO/openeo-python-driver/issues/243))

## 0.89.0

- Bump Werkzeug dependency to at least 2.3.8 (but below 3.0.0) for security issue ([#243](https://github.com/Open-EO/openeo-python-driver/issues/243))

## 0.88.0

- job metadata: remove un-official "file:nodata" field ([Open-EO/openeo-geopyspark-driver#588](https://github.com/Open-EO/openeo-geopyspark-driver/issues/588))

## 0.86.0

- Eliminate need to subclass `ConfigGetter`

## 0.85.0
- Expose mapping of job status to partial job status ([Open-EO/openeo-geopyspark-driver#644](https://github.com/Open-EO/openeo-geopyspark-driver/issues/644))

## 0.84.0

- Support GeoParquet output format for `aggregate_spatial` ([Open-EO/openeo-geopyspark-driver#623](https://github.com/Open-EO/openeo-geopyspark-driver/issues/623))

## 0.83.0

- Add `Processing.verify_for_synchronous_processing` API ([#248](https://github.com/Open-EO/openeo-python-driver/issues/248))

## 0.82.0

- Support EJR replacing ZkJobRegistry

## 0.81.0

- ~~Block sync request with too large extent. Use batch-job instead for those. ([Open-EO/openeo-geopyspark-driver#616](https://github.com/Open-EO/openeo-geopyspark-driver/issues/616))~~

## 0.80.0

- Add `User` argument to `GpsBatchJobs.create_job()`

## 0.79.0

- Disable basic auth support by default ([#90](https://github.com/Open-EO/openeo-python-driver/issues/90))

## 0.78.0

- `OpenEoBackendConfig`: make showing stack trace on `_load` configurable

## 0.77.4

- Flag `/openeo/1.2` API version as production ready ([#195](https://github.com/Open-EO/openeo-python-driver/issues/195))

## 0.77.2

- fixup "polygons" argument of "apply_polygon" ([#229](https://github.com/Open-EO/openeo-python-driver/issues/229))

## 0.76.1

- Attempt to workaround issue with in-place process graph modification
  and process registry process spec JSON (re)encoding ([Open-EO/openeo-geopyspark-driver#567](https://github.com/Open-EO/openeo-geopyspark-driver/issues/567))

## 0.76.0

- Add `OpenEoBackendConfig.deploy_env`

## 0.75.0

- Move `enable_basic_auth`/`enable_oidc_auth` to `OpenEoBackendConfig`

## 0.73.0

- add `ClientCredentials.from_credentials_string()`

## 0.72.3

- Improve request id logging when log collection failed ([Open-EO/openeo-geopyspark-driver#546](https://github.com/Open-EO/openeo-geopyspark-driver/issues/546))

## 0.72.2

- use `yymmdd` prefix in job/req ids for now

## 0.72.1

- Add access_token introspection result (when enabled) to `User.internal_auth_data`

## 0.72.0

- Start returning "OpenEO-Costs-experimental" header on synchronous processing responses
- Extract client credentials access token fetch logic from ElasticJobRegistry
  into `ClientCredentialsAccessTokenHelper` to make it reusable (e.g. for ETL API as well)
  ([Open-EO/openeo-geopyspark-driver#531](https://github.com/Open-EO/openeo-geopyspark-driver/issues/531))

## 0.71.0

- `OpenEoBackendImplementation.request_costs()`: add support for passing User object (related to [Open-EO/openeo-geopyspark-driver#531](https://github.com/Open-EO/openeo-geopyspark-driver/issues/531))

## 0.70.0

- Initial support for openeo-processes v2.0, when requesting version 1.2 of the openEO API ([#195](https://github.com/Open-EO/openeo-python-driver/issues/195))
- Drop support for 0.4 version of openeo-processes ([#47](https://github.com/Open-EO/openeo-python-driver/issues/47))


## 0.69.1

- Add backoff to ensure EJR deletion ([#163](https://github.com/Open-EO/openeo-python-driver/issues/163))


## 0.69.0

- Support job deletion in EJR ([#163](https://github.com/Open-EO/openeo-python-driver/issues/163))



<!-- end-of-changelog -->
