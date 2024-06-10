import logging
import sys
import re
from pathlib import Path
from typing import List, Union, Optional

import jinja2
import markdown

_log = logging.getLogger(__name__)

MULTI_PROJECT_CHANGELOG_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        .project-changelog {
            height: 32em;
            margin-left: 5em;
            padding: 0em 1em;
            overflow: scroll;
            border: 1px solid #ccc;
        }
    </style>
</head>
<body>
<div id="content">
    <h1>{{ title }}</h1>
    {% for project in projects %}
        <h2 id="{{ project.name | lower }}">{{ project.name }} {{ project.version }}</h2>
        {% if project.changelog_html %}
            <div class="project-changelog">{{ project.changelog_html }}</div>
        {% endif %}
    {% endfor %}
</div>
</body>
</html>
"""


def markdown_changelog_to_html(source: Union[str, Path]) -> str:
    """Convert changelog in Markdown format to HTML"""
    if isinstance(source, Path):
        source = source.read_text(encoding="utf8")

    # Get content between markers (when present)
    source = source.split("<!-- start-of-changelog -->")[-1].split("<!-- end-of-changelog -->")[0]

    html = markdown.markdown(source)

    # Simple allow-list based HTML sanitizing
    allowed_html_elements = {"h1", "h2", "h3", "ul", "ol", "li", "a", "code", "p", "b", "i"}
    html = re.sub(
        pattern=r"</?\s*([a-z0-9]+).*?>",
        repl=lambda m: m.group(0) if m.group(1) in allowed_html_elements else "",
        string=html,
    )

    return html


def multi_project_changelog(projects: List[dict], title: str = "Changelog") -> str:
    """
    Generate a multi-project changelog in HTML format

    Return this HTMl string as flask response with something like:

        return flask.make_response(html, {"Content-Type": "text/html"})

    :param projects: list of project dicts, with fields like "name", "version", "changelog_path"
    """
    projects = [p.copy() for p in projects]
    for project in projects:
        if project.get("changelog_path"):
            project["changelog_html"] = markdown_changelog_to_html(project.get("changelog_path"))

    html = jinja2.Template(MULTI_PROJECT_CHANGELOG_TEMPLATE).render(
        title=title,
        projects=projects,
    )
    return html


def get_changelog_path(
    data_files_dir: Optional[str], src_root: Optional[Path] = None, filename: str = "CHANGELOG.md"
) -> Union[Path, None]:
    """Get the path to the changelog file in the data files directory

    :param data_files_dir: `data_files` dir (as specified in setup.py/pyproject.toml) where CHANGELOG should be installed to
    :param src_root: source project root (in case of running from source)
    :param filename: Name of the changelog file
    """
    # Path of changelog when installed from wheel package
    if data_files_dir:
        installed_path = Path(sys.prefix) / data_files_dir / filename
        if installed_path.exists():
            return installed_path
        else:
            _log.warning(f"Failed to find changelog at {installed_path}")

    # Path of changelog when running from source
    if src_root:
        src_path = src_root / filename
        if src_path.exists():
            return src_path
