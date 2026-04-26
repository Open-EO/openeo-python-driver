"""
Process implementations sub-package.

Importing this package triggers registration of all process functions
into the process registries via their decorators.
"""
# Import all sub-modules to trigger @process / @process_registry decorators
from . import text, array, misc, terrain, ml, sar, geometry, temporal, udp, io, cubes  # noqa: F401