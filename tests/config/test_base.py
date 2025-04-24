import attrs
import pytest

from openeo_driver.config.base import (
    ConfigException,
    _ConfigBase,
    openeo_backend_config_class,
)


@pytest.mark.parametrize("strictness_mode", [None, "warn", "WARNING"])
def test_config_strictness_default_and_warn(monkeypatch, caplog, strictness_mode):
    if strictness_mode:
        monkeypatch.setenv("OPENEO_CONFIG_STRICTNESS_MODE", strictness_mode)

    caplog.set_level("WARNING")

    @openeo_backend_config_class
    class MyConfig(_ConfigBase):
        color = attrs.field(default="red", type=str)

    assert caplog.messages == []

    config = MyConfig(color="blue", size=42)

    assert "Ignoring invalid config arguments: {'size'}" in caplog.text

    assert config.color == "blue"
    with pytest.raises(AttributeError):
        _ = config.size


def test_config_strictness_ignore(monkeypatch, caplog):
    caplog.set_level("WARNING")

    monkeypatch.setenv("OPENEO_CONFIG_STRICTNESS_MODE", "ignore")

    @openeo_backend_config_class
    class MyConfig(_ConfigBase):
        color = attrs.field(default="red", type=str)

    assert caplog.messages == []

    config = MyConfig(color="blue", size=42)

    assert caplog.messages == []

    assert config.color == "blue"
    with pytest.raises(AttributeError):
        _ = config.size


def test_config_strictness_strict(monkeypatch, caplog):
    caplog.set_level("WARNING")

    monkeypatch.setenv("OPENEO_CONFIG_STRICTNESS_MODE", "strict")

    @openeo_backend_config_class
    class MyConfig(_ConfigBase):
        color = attrs.field(default="red", type=str)

    assert caplog.messages == []

    with pytest.raises(ConfigException, match="Invalid config arguments: {'size'}"):
        _ = MyConfig(color="blue", size=42)
