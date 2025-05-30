import re

from openeo_driver.server import build_backend_deploy_metadata
from openeo_driver.util.date_math import now_utc


def test_build_backend_deploy_metadata():
    data = build_backend_deploy_metadata(packages=["openeo", "openeo_driver", "foobarblerghbwop"])
    assert data["date"].startswith(now_utc().strftime("%Y-%m-%dT%H"))
    assert re.match(r"\d+\.\d+\.\d+", data["versions"]["openeo"])
    assert re.match(r"\d+\.\d+\.\d+", data["versions"]["openeo_driver"])
    assert data["versions"]["foobarblerghbwop"] == "n/a"
