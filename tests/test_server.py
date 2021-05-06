import re
from datetime import datetime

from openeo_driver.server import build_backend_deploy_metadata


def test_build_backend_deploy_metadata():
    data = build_backend_deploy_metadata(packages=["openeo", "openeo_driver", "foobarblerghbwop"])
    assert data["date"].startswith(datetime.utcnow().strftime("%Y-%m-%dT%H"))
    assert re.match(r"openeo \d+\.\d+\.\d+", data["versions"]["openeo"])
    assert re.match(r"openeo-driver \d+\.\d+\.\d+", data["versions"]["openeo_driver"])
    assert data["versions"]["foobarblerghbwop"] == "n/a"
