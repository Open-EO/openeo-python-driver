import random
import textwrap

import attrs.exceptions
import pytest

from openeo_driver.config import (
    OpenEoBackendConfig,
    get_backend_config,
    ConfigException,
)
from openeo_driver.config.load import load_from_py_file


def test_config_immutable():
    conf = OpenEoBackendConfig(id="dontchangeme!")
    assert conf.id == "dontchangeme!"
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        conf.id = "let's try?"
    assert conf.id == "dontchangeme!"


def test_load_from_py_file_default(tmp_path):
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

    config = load_from_py_file(path)
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


def test_get_backend_config(monkeypatch, tmp_path):
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

    monkeypatch.setenv("_OPENEO_BACKEND_CONFIG", str(path))

    get_backend_config.flush()
    config1 = get_backend_config()
    assert isinstance(config1, OpenEoBackendConfig)

    config2 = get_backend_config()
    assert config2 is config1

    get_backend_config.flush()
    config3 = get_backend_config()
    assert not (config3 is config1)
