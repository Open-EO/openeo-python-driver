import pytest

from openeo_driver.util.pgparsing import SingleRunUDFProcessGraph, NotASingleRunUDFProcessGraph


class TestSingleRunUDFProcessGraph:
    def test_parse_basic(self):
        pg = {
            "runudf1": {
                "process_id": "run_udf",
                "arguments": {
                    "data": {"from_parameter": "data"},
                    "udf": "print('Hello world')",
                    "runtime": "Python",
                },
                "result": True,
            }
        }
        run_udf = SingleRunUDFProcessGraph.parse(pg)
        assert run_udf.data == {"from_parameter": "data"}
        assert run_udf.udf == "print('Hello world')"
        assert run_udf.runtime == "Python"
        assert run_udf.version is None
        assert run_udf.context == {}

    @pytest.mark.parametrize(
        "pg",
        [
            {
                "runudf1": {
                    "process_id": "run_udffffffffffffffff",
                    "arguments": {"data": {"from_parameter": "data"}, "udf": "x = 4", "runtime": "Python"},
                    "result": True,
                }
            },
            {
                "runudf1": {
                    "process_id": "run_udf",
                    "arguments": {"udf": "x = 4", "runtime": "Python"},
                    "result": True,
                }
            },
            {
                "runudf1": {
                    "process_id": "run_udf",
                    "arguments": {"data": {"from_parameter": "data"}, "runtime": "Python"},
                    "result": True,
                }
            },
            {
                "runudf1": {
                    "process_id": "run_udf",
                    "arguments": {"data": {"from_parameter": "data"}, "udf": "x = 4"},
                    "result": True,
                }
            },
            {
                "runudf1": {
                    "process_id": "run_udf",
                    "arguments": {"data": {"from_parameter": "data"}, "udf": "x = 4", "runtime": "Python"},
                }
            },
            {
                "runudf1": {
                    "process_id": "run_udf",
                    "arguments": {"data": {"from_parameter": "data"}, "udf": "x = 4", "runtime": "Python"},
                    "result": True,
                },
                "runudf2": {
                    "process_id": "run_udf",
                    "arguments": {"data": {"from_parameter": "data"}, "udf": "x = 4", "runtime": "Python"},
                },
            },
        ],
    )
    def test_parse_invalid(self, pg):
        with pytest.raises(NotASingleRunUDFProcessGraph):
            _ = SingleRunUDFProcessGraph.parse(pg)
