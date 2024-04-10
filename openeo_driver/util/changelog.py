import re
from pathlib import Path
from typing import List, Union

import jinja2
import markdown

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
        <div class="project-changelog">{{ project.changelog_html }}</div>
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

    :param projects: list of project dicts, with fields like "name", "version", "changelog_path"
    """
    projects = [p.copy() for p in projects]
    for project in projects:
        if "changelog_path" in project:
            project["changelog_html"] = markdown_changelog_to_html(project["changelog_path"])

    html = jinja2.Template(MULTI_PROJECT_CHANGELOG_TEMPLATE).render(
        title=title,
        projects=projects,
    )
    return html
