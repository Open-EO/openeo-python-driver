from openeo_driver.errors import OpenEOApiException
from openeo_driver.processgraph import get_process_definition_from_url

import pytest


class TestProcessDefinitionFromUrl:
    PROCESS_ADD35 = {
        "id": "add35",
        "process_graph": {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}},
        "parameters": [],
        "returns": {"schema": {"type": "number"}},
    }

    PROCESS_ADD3PARAM = {
        "id": "add3param",
        "process_graph": {
            "add": {"process_id": "add", "arguments": {"x": 3, "y": {"from_parameter": "delta"}}, "result": True}
        },
        "parameters": [
            {"name": "delta", "schema": {"type": "number", "optional": True, "default": 1}},
        ],
        "returns": {"schema": {"type": "number"}},
    }

    def test_get_process_definition_from_url_single(self, requests_mock):
        requests_mock.get("https://share.test/add3param.json", json=self.PROCESS_ADD3PARAM)

        pd = get_process_definition_from_url("add3param", "https://share.test/add3param.json")
        assert pd.id == "add3param"
        assert pd.process_graph == {
            "add": {"process_id": "add", "arguments": {"x": 3, "y": {"from_parameter": "delta"}}, "result": True},
        }
        assert pd.parameters == [{"name": "delta", "schema": {"type": "number", "optional": True, "default": 1}}]
        assert pd.returns == {"schema": {"type": "number"}}

    def test_get_process_definition_from_url_listing(self, requests_mock):
        requests_mock.get(
            "https://share.test/processes/",
            json={
                "processes": [
                    self.PROCESS_ADD35,
                    self.PROCESS_ADD3PARAM,
                ],
                "links": [],
            },
        )

        pd = get_process_definition_from_url("add3param", "https://share.test/processes/")
        assert pd.id == "add3param"
        assert pd.process_graph == {
            "add": {"process_id": "add", "arguments": {"x": 3, "y": {"from_parameter": "delta"}}, "result": True},
        }
        assert pd.parameters == [{"name": "delta", "schema": {"type": "number", "optional": True, "default": 1}}]
        assert pd.returns == {"schema": {"type": "number"}}
