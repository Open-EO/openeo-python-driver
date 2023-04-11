from typing import Set

import requests
import requests.adapters


def requests_with_retry(
    total: int = 3,
    backoff_factor: float = 1,
    status_forcelist: Set[int] = frozenset([429, 500, 502, 503, 504]),
    **kwargs,
) -> requests.Session:
    """
    Create a `requests.Session` with automatic retrying

    Inspiration and references:
    - https://requests.readthedocs.io/en/latest/api/#requests.adapters.HTTPAdapter
    - https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#urllib3.util.Retry
    - https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/#retry-on-failure
    """
    session = requests.Session()
    retry = requests.adapters.Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        **kwargs,
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
