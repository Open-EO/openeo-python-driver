import textwrap

from openeo_driver.util.changelog import (
    markdown_changelog_to_html,
    multi_project_changelog,
)


def test_markdown_changelog_to_html_basic():
    markdown = textwrap.dedent(
        """
        ## 1.2.3
        - Add more oomph
        """
    )
    html = markdown_changelog_to_html(markdown)
    assert html.startswith("<h2>1.2.3</h2>")
    assert "<li>Add more oomph</li>" in html


def test_markdown_changelog_to_html_markers():
    markdown = textwrap.dedent(
        """
        Describe changes below
        <!-- start-of-changelog -->
        ## 1.2.3
        - Add more oomph
        <!-- end-of-changelog -->
        kthxbye
        """
    )
    html = markdown_changelog_to_html(markdown)
    assert html.startswith("<h2>1.2.3</h2>")
    assert "<li>Add more oomph</li>" in html
    assert "Describe changes below" not in html
    assert "kthxbye" not in html


def test_markdown_changelog_to_html_sanitizing():
    markdown = textwrap.dedent(
        """
        ## 1.2.3
        <script src="https://evil.com/evil.js"></script>
        - Add more oomph
        """
    )
    html = markdown_changelog_to_html(markdown)
    assert "script" not in html
    assert "evil" not in html
    assert "<li>Add more oomph</li>" in html


def test_multi_project_changelog(tmp_path):
    changelog_path = tmp_path / "changelog.md"
    changelog_path.write_text(
        textwrap.dedent(
            """
            ## 1.0.2
            - Fix ipsum
            """
        )
    )
    html = multi_project_changelog(
        [
            {"name": "Lorem", "version": "1.2.3", "changelog_path": changelog_path},
            {"name": "IpsumLib", "version": "24.3"},
        ]
    )
    assert html.startswith("<!DOCTYPE html>\n<html>")
    assert '<h2 id="lorem">Lorem 1.2.3</h2>' in html
    assert "<h2>1.0.2</h2>" in html
    assert "<li>Fix ipsum</li>" in html
    assert '<h2 id="ipsumlib">IpsumLib 24.3</h2>' in html
