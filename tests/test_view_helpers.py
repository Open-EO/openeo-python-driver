import datetime

import flask
import pytest

from openeo_driver.util.view_helpers import cache_control


class TestCacheControl:
    @pytest.fixture(autouse=True)
    def set_time(self, time_machine):
        time_machine.move_to("2022-06-01T15:50:00", tick=False)

    def test_default(self, ):
        app = flask.Flask(__name__)

        @app.route("/hello")
        @cache_control()
        def hello():
            return flask.jsonify(hello="world")

        with app.test_client() as client:
            resp = client.get("/hello")
        assert "Cache-Control" not in resp.headers
        assert "Expires" not in resp.headers

    @pytest.mark.parametrize("max_age", [
        123,
        datetime.timedelta(minutes=2, seconds=3),
    ])
    def test_max_age(self, max_age):
        app = flask.Flask(__name__)

        @app.route("/hello")
        @cache_control(max_age=max_age)
        def hello():
            return flask.jsonify(hello="world")

        with app.test_client() as client:
            resp = client.get("/hello")
        assert resp.headers["Cache-Control"] == "max-age=123"
        assert resp.headers["Expires"] == "Wed, 01 Jun 2022 15:52:03 GMT"

    def test_multiple_views(self):
        app = flask.Flask(__name__)

        @app.route("/hello")
        @cache_control(max_age=123)
        def hello():
            return flask.jsonify(hello="world")

        @app.route("/hi")
        @cache_control(max_age=1234)
        def hi():
            return flask.jsonify(hi="world")

        @app.route("/bye")
        def bye():
            return flask.jsonify(bye="world")

        with app.test_client() as client:
            resp = client.get("/hello")
            assert resp.headers["Cache-Control"] == "max-age=123"
            resp = client.get("/hi")
            assert resp.headers["Cache-Control"] == "max-age=1234"
            resp = client.get("/bye")
            assert "Cache-Control" not in resp.headers

    @pytest.mark.parametrize(["status", "caching"], [
        (200, True),
        (302, False),
        (404, False),
        (403, False),
        (500, False),
    ])
    def test_no_cache_on_failure(self, status, caching):
        app = flask.Flask(__name__)

        @app.route("/hello")
        @cache_control(max_age=123)
        def hello():
            return flask.Response(response="hello", status=status)

        with app.test_client() as client:
            resp = client.get("/hello")

        if caching:
            assert resp.headers["Cache-Control"] == "max-age=123"
            assert resp.headers["Expires"] == "Wed, 01 Jun 2022 15:52:03 GMT"
        else:
            assert "Cache-Control" not in resp.headers
            assert "Expires" not in resp.headers

    def test_multiple_directives(self):
        app = flask.Flask(__name__)

        @app.route("/hello")
        @cache_control(max_age=123, no_cache=True, public=True, must_revalidate=True)
        def hello():
            return flask.jsonify(hello="world")

        with app.test_client() as client:
            resp = client.get("/hello")
        assert resp.headers["Cache-Control"] == "max-age=123, no-cache, public, must-revalidate"
        assert resp.headers["Expires"] == "Wed, 01 Jun 2022 15:52:03 GMT"
