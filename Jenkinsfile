#!/usr/bin/env groovy

/*
    This Jenkinsfile is used to provide snapshot builds using the VITO CI system.
    This Jenkinsfile uses the internal "biggeo/jenkinslib.git" library.
    Information about the pythonPipeline method can be found in pythonPipeline.groovy
*/

@Library('lib')_

pythonPipeline {
  package_name = 'openeo_driver'
  wipeout_workspace = true
  python_version = ["3.8"]
  downstream_job = 'openEO/openeo-integrationtests'
  wheel_repo = 'python-openeo'
  extras_require = 'dev'
  upload_dev_wheels = false
  pep440 = true
  custom_test_image = 'vito-docker.artifactory.vgt.vito.be/centos8-spark-py-openeo:3.2.0'
  extra_env_variables = [
    /* Set pytest `basetemp` inside Jenkins workspace. (Note: this is intentionally Jenkins specific, instead of a global pytest.ini thing.) */
    "PYTEST_DEBUG_TEMPROOT=pytest-tmp",
  ]
  pre_test_script = 'pre_test.sh'
}
