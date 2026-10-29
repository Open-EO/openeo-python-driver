#!/usr/bin/env groovy

/*
    This Jenkinsfile is used to run tests with Python 3.11 using the VITO CI system.
    This Jenkinsfile uses the internal "biggeo/jenkinslib.git" library.
    Information about the pythonPipeline method can be found in pythonPipeline.groovy
*/

@Library('lib')_

pythonPipeline {
  package_name = 'openeo_driver'
  wipeout_workspace = true
  python_version = ["3.11"]
  downstream_job = 'openEO/trigger-python311-build'
  extras_require = 'dev'
  upload_dev_wheels = false
  build_wheel = true
  extra_env_variables = [
    /* Set pytest `basetemp` inside Jenkins workspace. (Note: this is intentionally Jenkins specific, instead of a global pytest.ini thing.) */
    "PYTEST_DEBUG_TEMPROOT=pytest-tmp",
  ]
  pre_test_script = 'pre_test.sh'
  enable_uv = true
  enable_caching = true
  wheel_repo = 'openeo-pypi-local'
  wheel_repo_dev = 'openeo-pypi-local'
  artifactory_server = 'sas'
}
