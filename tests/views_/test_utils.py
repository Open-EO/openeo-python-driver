from openeo_driver.views_.utils import add_link_by_rel


def test_add_link_by_rel_append_mode() -> None:
    links = [{"rel": "foo", "href": "https://base.test/foo"}]

    # Generic case: different rel, default mode (append)
    assert add_link_by_rel(links, link={"rel": "bar", "href": "https://bar.test/"}) == [
        {"rel": "foo", "href": "https://base.test/foo"},
        {"rel": "bar", "href": "https://bar.test/"},
    ]

    # Same rel
    assert add_link_by_rel(links, link={"rel": "foo", "href": "https://foo.test/"}) == [
        {"rel": "foo", "href": "https://base.test/foo"},
        {"rel": "foo", "href": "https://foo.test/"},
    ]
    # Same rel, append mode
    assert add_link_by_rel(links, link={"rel": "foo", "href": "https://foo.test/"}, mode="append") == [
        {"rel": "foo", "href": "https://base.test/foo"},
        {"rel": "foo", "href": "https://foo.test/"},
    ]
    assert links == [{"rel": "foo", "href": "https://base.test/foo"}]


def test_add_link_by_rel_fallback_mode() -> None:
    links = [{"rel": "foo", "href": "https://base.test/foo"}]

    assert add_link_by_rel(links, link={"rel": "foo", "href": "https://foo.test/"}, mode="fallback") == [
        {"rel": "foo", "href": "https://base.test/foo"},
    ]
    assert add_link_by_rel(links, link={"rel": "bar", "href": "https://foo.test/"}, mode="fallback") == [
        {"rel": "foo", "href": "https://base.test/foo"},
        {"rel": "bar", "href": "https://foo.test/"},
    ]

    assert links == [{"rel": "foo", "href": "https://base.test/foo"}]


def test_add_link_by_rel_overwrite_mode() -> None:
    links = [{"rel": "foo", "href": "https://base.test/foo"}]

    assert add_link_by_rel(links, link={"rel": "foo", "href": "https://foo.test/"}, mode="overwrite") == [
        {"rel": "foo", "href": "https://foo.test/"},
    ]
    assert add_link_by_rel(links, link={"rel": "bar", "href": "https://foo.test/"}, mode="overwrite") == [
        {"rel": "foo", "href": "https://base.test/foo"},
        {"rel": "bar", "href": "https://foo.test/"},
    ]

    assert links == [{"rel": "foo", "href": "https://base.test/foo"}]
