import pytest

from openeo_driver.config.load import ConfigGetter, ConfigException


@pytest.fixture
def simple_dict_config(monkeypatch, tmp_path):
    """
    Set up a config file, just containing a dict as 'config'.
    """
    path = tmp_path / "config.py"
    path.write_text('config = {"foo": "bar"}\n')
    monkeypatch.setenv(ConfigGetter.OPENEO_BACKEND_CONFIG, str(path))


def test_expected_class_legacy(simple_dict_config):
    class DictConfigGetter(ConfigGetter):
        expected_class = dict

    class ListConfigGetter(ConfigGetter):
        expected_class = list

    config = DictConfigGetter().get()
    assert config == {"foo": "bar"}

    with pytest.raises(ConfigException, match="Expected list but got dict"):
        _ = ListConfigGetter().get()


def test_expected_class_init(simple_dict_config):
    config = ConfigGetter(expected_class=dict).get()
    assert config == {"foo": "bar"}

    with pytest.raises(ConfigException, match="Expected list but got dict"):
        _ = ConfigGetter(expected_class=list).get()
