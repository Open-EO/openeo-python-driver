"""
Utilities for (lazy) loading of config
"""
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union, ContextManager

try:
    # Use `importlib_resources` instead of stdlib `importlib.resources`
    # to have backported fix in Python<3.10 for https://bugs.python.org/issue44137
    import importlib_resources
except ImportError:
    import importlib.resources

    importlib_resources = importlib.resources

from openeo_driver.config import OpenEoBackendConfig, ConfigException


_log = logging.getLogger(__name__)


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

    # Environment variable to point to config file
    OPENEO_BACKEND_CONFIG = "_OPENEO_BACKEND_CONFIG"

    expected_class = OpenEoBackendConfig

    def __init__(self):
        self._config: Optional[OpenEoBackendConfig] = None

    def __call__(self, force_reload: bool = False) -> OpenEoBackendConfig:
        """Syntactic sugar to lazy load the config with function call."""
        return self.get(force_reload=force_reload)

    def get(self, force_reload: bool = False) -> OpenEoBackendConfig:
        """Lazy load the config."""
        if self._config is None or force_reload:
            self._config = self._load()
        return self._config

    def _default_config(self) -> ContextManager[Path]:
        """
        Default config file (as a context manager to allow it to be an ephemeral resource).
        """
        return importlib_resources.path("openeo_driver.config", "default.py")

    def _load(self) -> OpenEoBackendConfig:
        """Load the config from config file."""
        with self._default_config() as default_config:
            config_path = os.environ.get(self.OPENEO_BACKEND_CONFIG, default_config)
            config = load_from_py_file(path=config_path, variable="config", expected_class=self.expected_class)
        return config

    def flush(self):
        """Flush the config, to force a reload on next get."""
        self._config = None


get_backend_config = ConfigGetter()