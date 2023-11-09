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


## In progress

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
