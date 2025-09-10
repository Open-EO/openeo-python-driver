from openeo_driver.errors import ProcessGraphInvalidException
from openeo_driver.processgraph import (
    get_process_definition_from_url,
    extract_default_job_options_from_process_graph,
    ProcessGraphFlatDict,
)

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


def test_extract_default_job_options_from_process_graph(requests_mock):
    requests_mock.get(
        "https://share.test/add3.json",
        json={
            "id": "add3",
            "process_graph": {
                "add": {"process_id": "add", "arguments": {"x": {"from_parameter": "x"}, "y": 3}, "result": True}
            },
            "parameters": [
                {"name": "x", "schema": {"type": "number"}},
            ],
            "returns": {"schema": {"type": "number"}},
            "default_job_options": {
                "memory": "2GB",
                "cpu": "yellow",
            },
            "default_synchronous_options": {
                "cpu": "green",
            },
        },
    )

    pg = ProcessGraphFlatDict(
        {
            "add12": {"process_id": "add", "arguments": {"x": 1, "y": 2}},
            "add3": {
                "process_id": "add3",
                "namespace": "https://share.test/add3.json",
                "arguments": {"x": {"from_node": "add12"}},
                "result": True,
            },
        }
    )
    assert extract_default_job_options_from_process_graph(pg, processing_mode="batch_job") == {
        "memory": "2GB",
        "cpu": "yellow",
    }
    assert extract_default_job_options_from_process_graph(pg, processing_mode="synchronous") == {
        "cpu": "green",
    }


@pytest.mark.parametrize(
    "pg",
    [
        {"garbage": "yez"},
        {"almost": {"procezz_id": "add", "arguments": {"x": "nope"}}},
    ],
)
def test_extract_default_job_options_from_process_graph_garbage(pg):
    with pytest.raises(ProcessGraphInvalidException):
        extract_default_job_options_from_process_graph(ProcessGraphFlatDict(pg))
