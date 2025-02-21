import inspect
import logging
import os
from typing import Type, Set, Dict, Any

import attrs

_log = logging.getLogger(__name__)


class ConfigException(ValueError):
    pass


# Reusable decorator for openEO backend config classes
openeo_backend_config_class = attrs.frozen(
    # Note: `kw_only=True` enforces "kwargs" based construction (which is good for readability/maintainability)
    # and allows defining mandatory fields (fields without default) after optional fields.
    kw_only=True,
    # Disable automatically adding `__init__` to preserve the one from `_ConfigBase`
    init=False,
)


@openeo_backend_config_class
class _ConfigBase:
    """
    Base class for OpenEoBackendConfig and subclasses,
    with some generic internals
    """

    def __init__(self, **kwargs):
        # Custom __init__ to filter out unknown kwargs
        # and depending on strictness mode: ignore them, fail hard, or log a warning.

        supported: Set[str] = {a.alias for a in attrs.fields(type(self))}
        valid: Dict[str, Any] = {}
        invalid: Set[str] = set()
        for k, v in kwargs.items():
            if k in supported:
                valid[k] = v
            else:
                invalid.add(k)

        if invalid:
            strictness_mode = os.environ.get("OPENEO_CONFIG_STRICTNESS_MODE")
            if strictness_mode == "strict":
                raise ConfigException(f"Invalid config arguments: {invalid}")
            elif strictness_mode == "ignore":
                pass
            else:
                # Warn by default.
                _log.warning(f"Ignoring invalid config arguments: {invalid}")

        self.__attrs_init__(**valid)




def check_config_definition(config_class: Type[_ConfigBase]):
    """
    Verify that the config class definition is correct (e.g. all fields have type annotations).
    """
    annotated = set(k for cls in config_class.__mro__ for k in cls.__dict__.get("__annotations__", {}).keys())
    members = set(k for k, v in inspect.getmembers(config_class) if not k.startswith("_"))

    if annotated != members:
        missing_annotations = members.difference(annotated)
        raise ConfigException(f"{config_class.__name__}: fields without type annotation: {missing_annotations}.")
