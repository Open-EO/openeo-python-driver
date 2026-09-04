from typing import Iterable, List, Literal


def add_link_by_rel(
    links: Iterable[dict], *, link: dict, mode: Literal["append", "fallback", "overwrite"] = "append"
) -> List[dict]:
    """
    Add a link to the given collection of links, producing a new list links,
    taking care of the "rel" attribute, e.g. to avoid duplicates:
    - "append": always append the new link to the list
    - "fallback": only append the new link if no link with the same rel already exists
    - "overwrite": remove any existing links with the same rel before appending the new link
    """

    # Work on a copy
    links = list(links)

    if mode == "append":
        links += [link]
    elif mode == "fallback":
        if not any(l.get("rel") == link.get("rel") for l in links):
            links.append(link)
    elif mode == "overwrite":
        links = [l for l in links if l.get("rel") != link.get("rel")] + [link]
    else:
        raise ValueError(f"Invalid {mode=}")
    return links
