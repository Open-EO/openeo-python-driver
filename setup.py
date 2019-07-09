from setuptools import setup, find_packages
import os
import datetime

# Load the openeo version info.
#
# Note that we cannot simply import the module, since dependencies listed
# in setup() will very likely not be installed yet when setup.py run.
#
# See:
#   https://packaging.python.org/guides/single-sourcing-package-version

__version__ = None
date = datetime.datetime.today().strftime('%Y%m%d')

with open('openeo_driver/_version.py') as fp:
    exec(fp.read())

if os.environ.get('BUILD_NUMBER') and os.environ.get('BRANCH_NAME'):
    if os.environ.get('BRANCH_NAME') == 'develop':
        version = __version__ + '.' + date + '.' + os.environ['BUILD_NUMBER']
    else:
        version = __version__ + '.' + date + '.' + os.environ['BUILD_NUMBER'] + '+' + os.environ['BRANCH_NAME']
else:
    version = __version__

setup(
    name='openeo_driver',
    version=version,
    packages=find_packages(include=['openeo*']),
    include_package_data=True,
    setup_requires=['pytest-runner'],
    tests_require=['pytest','mock','requests-mock'],
    install_requires=[
        'flask',
        'openeo',
        'gunicorn',
        'shapely'
    ],
)
