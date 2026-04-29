"""mkdocs hooks: rewrite paths in content mirrored from tile-ai/TileOPs.

Upstream files reference repo-relative paths (`../../tileops/...`,
`../../.claude/skills/<name>/SKILL.md`, `../../docs/design/...`) that
don't resolve once mirrored into this site. We rewrite them at the
`on_page_markdown` stage, after the include-markdown plugin has pulled
content in.
"""
from __future__ import annotations

import re

UPSTREAM_BLOB = "https://github.com/tile-ai/TileOPs/blob/main"

_SKILL_LINK = re.compile(r"(?:\.\./)+\.claude/skills/([\w-]+)/SKILL\.md")
_SKILL_DESIGN_REF = re.compile(r"\.\./\.\./\.\./docs/design/")
_SKILL_SIBLING = re.compile(r"\.\./([\w-]+)/SKILL\.md")
_DESIGN_REPO_PATH = re.compile(r"\.\./\.\./([\w./-]+)")


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_path.replace("\\", "/")

    if src.startswith("design/"):
        # Upstream design docs link to source files via ../../<repo path>.
        # Redirect those to GitHub so they resolve from the published site.
        markdown = _DESIGN_REPO_PATH.sub(rf"{UPSTREAM_BLOB}/\1", markdown)

    elif src.startswith("skills/"):
        # Catalog and overview embed sections that link to .claude/skills/<name>/SKILL.md
        # — redirect to the local guide page.
        markdown = _SKILL_LINK.sub(r"\1.md", markdown)
        # Per-skill guide pages embed SKILL.md, which links to design docs via
        # ../../../docs/design/ and to sibling skills via ../<name>/SKILL.md.
        markdown = _SKILL_DESIGN_REF.sub("../design/", markdown)
        markdown = _SKILL_SIBLING.sub(r"\1.md", markdown)

    return markdown
