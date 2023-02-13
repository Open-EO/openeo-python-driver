import datetime
import functools

import flask


def _utcnow() -> datetime.datetime:
    # Allow patching utcnow for unit testing
    # TODO: just start using `time_machine` module for time mocking
    return datetime.datetime.utcnow()


def cache_control(
        max_age=None, no_cache=None, no_store=None,
        public=None, private=None, must_revalidate=None,
):
    """
    Parameterized decorator for view functions to set `Cache-Control` headers on the response.
    """
    if isinstance(max_age, datetime.timedelta):
        max_age = int(max_age.total_seconds())

    settings = dict(
        max_age=max_age, no_cache=no_cache, no_store=no_store,
        public=public, private=private, must_revalidate=must_revalidate,
    )
    settings = {key: value for key, value in settings.items() if value is not None}

    def add_cache_control_headers(response: flask.Response) -> flask.Response:
        # TODO: option to take status code into account
        if 200 <= response.status_code < 300:
            for key, value in settings.items():
                setattr(response.cache_control, key, value)
            if max_age is not None:
                response.expires = _utcnow() + datetime.timedelta(seconds=max_age)
        return response

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            flask.after_this_request(add_cache_control_headers)
            return func(*args, **kwargs)

        return wrapped

    return decorator
