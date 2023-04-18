"""
Utilities for (lazy) loading of config
"""
import importlib.resources
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from openeo_driver.config.config import ConfigException

_log = logging.getLogger(__name__)

from openeo_driver.config import OpenEoBackendConfig


def load_from_py_file(
    path: Union[str, Path],
    variable: str = "config",
    expected_class: Optional[type] = OpenEoBackendConfig,
) -> Any:
    """Load a config value from a Python file."""
    path = Path(path)
    _log.info(
        f"Loading configuration from Python file {path!r} (variable {variable!r})"
    )

    # Based on flask's Config.from_pyfile
    with path.open(mode="rb") as f:
        code = compile(f.read(), path, "exec")
    globals = {"__file__": str(path)}
    exec(code, globals)

    try:
        config = globals[variable]
    except KeyError:
        raise ConfigException(
            f"No variable {variable!r} found in config file {path!r}"
        ) from None

    if expected_class:
        if not isinstance(config, expected_class):
            raise ConfigException(
                f"Expected {expected_class.__name__} but got {type(config).__name__}"
            )
    return config


class ConfigGetter:
    """Config loader, with lazy-loading and flushing."""

    def __init__(self):
        self._config: Optional[OpenEoBackendConfig] = None

    def __call__(self, force_reload: bool = False) -> OpenEoBackendConfig:
        if self._config is None or force_reload:
            self._config = self._load()
        return self._config

    def _load(self) -> OpenEoBackendConfig:
        with importlib.resources.path(
            "openeo_driver.config", "default.py"
        ) as default_config:
            config_path = os.environ.get(
                "OPENEO_BACKEND_CONFIG",
                default_config,
            )
            config = load_from_py_file(
                path=config_path, variable="config", expected_class=OpenEoBackendConfig
            )
            return config

    def flush(self):
        self._config = None


get_backend_config = ConfigGetter()
