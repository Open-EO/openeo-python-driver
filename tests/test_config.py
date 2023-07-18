import json
import random
import textwrap
from pathlib import Path

import attrs.exceptions
import pytest

import openeo_driver.config.load
from openeo_driver.config import ConfigException, OpenEoBackendConfig, get_backend_config
from openeo_driver.config.load import load_from_py_file
from .conftest import enhanced_logging


def test_config_immutable():
    conf = OpenEoBackendConfig(id="dontchangeme!")
    assert conf.id == "dontchangeme!"
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        conf.id = "let's try?"
    assert conf.id == "dontchangeme!"


@pytest.mark.parametrize("path_type", [str, Path])
def test_load_from_py_file_default(tmp_path, path_type):
    path = tmp_path / "myconfig.py"
    cid = f"config-{random.randint(1, 100000)}"

    content = f"""
        from openeo_driver.config import OpenEoBackendConfig
        cid = {cid!r}
        config = OpenEoBackendConfig(
            id=cid
        )
    """
    content = textwrap.dedent(content)
    path.write_text(content)

    config = load_from_py_file(path_type(path))
    assert isinstance(config, OpenEoBackendConfig)
    assert config.id == cid


def test_load_from_py_file_custom(tmp_path):
    path = tmp_path / "myconfig.py"
    cid = f"config-{random.randint(1, 100000)}"
    path.write_text(f'konff = ("hello world", {cid!r}, list(range(3)))')
    config = load_from_py_file(path, variable="konff", expected_class=tuple)
    assert isinstance(config, tuple)
    assert config == ("hello world", cid, [0, 1, 2])


def test_load_from_py_file_wrong_type(tmp_path):
    path = tmp_path / "myconfig.py"
    path.write_text(f"config = [3, 5, 8]")
    with pytest.raises(
        ConfigException, match="Expected OpenEoBackendConfig but got list"
    ):
        _ = load_from_py_file(path)


@pytest.fixture
def backend_config_flush():
    """Generate callable to flush the backend_config and automatically flush at start and end of test"""

    def flush():
        openeo_driver.config.load._backend_config_getter.flush()

    flush()
    yield flush
    flush()


def test_default_config(backend_config_flush, monkeypatch):
    monkeypatch.delenv("OPENEO_BACKEND_CONFIG")
    config = get_backend_config()
    assert config.id == "default"


def test_get_backend_config_lazy_cache(monkeypatch, tmp_path, backend_config_flush):
    path = tmp_path / "myconfig.py"
    content = """
        import random
        from openeo_driver.config import OpenEoBackendConfig
        config = OpenEoBackendConfig(
            id=f"config-{random.randint(1, 100000)}"
        )
    """
    content = textwrap.dedent(content)
    path.write_text(content)

    monkeypatch.setenv("OPENEO_BACKEND_CONFIG", str(path))

    backend_config_flush()
    config1 = get_backend_config()
    assert isinstance(config1, OpenEoBackendConfig)

    config2 = get_backend_config()
    assert config2 is config1

    backend_config_flush()
    config3 = get_backend_config()
    assert not (config3 is config1)

    # Remove config file
    path.unlink()
    config4 = get_backend_config()
    assert config4 is config3

    # Force reload should fail
    with pytest.raises(FileNotFoundError):
        _ = get_backend_config(force_reload=True)


def test_config_load_info_log(backend_config_flush):
    with enhanced_logging(json=True, context="test") as logs:
        _ = get_backend_config()

    logs = [json.loads(l) for l in logs.getvalue().strip().split("\n")]
    (log,) = [log for log in logs if log["message"].startswith("Loaded config")]
    assert "test_config_load_info_log" in log["stack_info"]


def test_get_backend_config_not_found(monkeypatch, tmp_path, backend_config_flush):
    monkeypatch.setenv("OPENEO_BACKEND_CONFIG", str(tmp_path / "nonexistent.py"))
    backend_config_flush()
    with pytest.raises(FileNotFoundError):
        _ = get_backend_config()


def test_kw_only():
    with pytest.raises(TypeError, match="takes 1 positional argument but 3 were given"):
        OpenEoBackendConfig(123, [])


def test_add_mandatory_fields():
    @attrs.frozen(kw_only=True)
    class MyConfig(OpenEoBackendConfig):
        color: str = "red"
        set_this_or_die: int

    with pytest.raises(TypeError, match="missing.*required.*argument.*set_this_or_die"):
        _ = MyConfig()

    conf = MyConfig(set_this_or_die=4)
    assert conf.set_this_or_die == 4


@pytest.mark.parametrize(
    ["backend_config_overrides", "expected_id"],
    [
        (None, "dummy"),
        ({}, "dummy"),
        ({"id": "overridden!"}, "overridden!"),
    ],
)
def test_pytest_override_context(backend_config, backend_config_overrides, expected_id):
    config = get_backend_config()
    assert config.id == expected_id
