"""mkdocs hooks: mirrored-content path rewrites, the Benchmarks nav, and the
untranslated-page notice.

* Mirrored design docs carry repo-relative paths (`../../tileops/...`) that do
  not resolve here; `on_page_markdown` rewrites them after include-markdown has
  pulled the content in.
* The Benchmarks pages are generated at deploy time and cannot be listed in
  `mkdocs.yml`; `on_config` expands that one nav entry to whatever the renderer
  produced.
* mkdocs-static-i18n serves the default-language page where a locale has no
  translation; `on_page_markdown` prepends a notice, so a fallback page reads as
  a translation still to come rather than a broken one.
* A blog post's first paragraph is its subtitle; `on_page_content` marks it for
  extra.css, so the post's Markdown carries no styling.
* A section index lists its pages as a plain Markdown list; `on_page_content`
  marks those lists for extra.css, which draws each item as a card.
* A page merged into another leaves its old URL behind; `on_post_build` writes
  a redirect there, so published links keep working.
* The stylesheet's URL carries a hash of its content, set in `on_config`, so a
  browser holding an older copy fetches the new one as soon as it changes.
"""
from __future__ import annotations

import hashlib
import os
import re

UPSTREAM_BLOB = "https://github.com/tile-ai/TileOPs/blob/main"

_DESIGN_REPO_PATH = re.compile(r"\.\./\.\./([\w./-]+)")

# Keyed by the locale being built, not by the locale of the content shown.
# A plain block, not an admonition: admonitions are styled as technical asides
# here, and a translation-status note should not compete with them. The class
# also tells extra.css to skip the Chinese type metrics on this page, since the
# body text it wraps is English.
_FALLBACK_NOTICE = {
    "zh": '<div class="locale-notice" translate="no">'
          "<strong>本页暂无中文版。</strong>以下为英文原文。</div>",
}


def _fallback_notice(page):
    """Notice for a page served from another locale, or "" when translated.

    i18n sets `locale` to the language of the file it picked and
    `locale_alternate_of` to the language currently being built; they differ
    exactly on a fallback.
    """
    file = page.file
    building = getattr(file, "locale_alternate_of", None)
    if building is None or getattr(file, "locale", None) == building:
        return ""
    return _FALLBACK_NOTICE.get(building, "")


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_path.replace("\\", "/")

    if src.startswith("design/"):
        # Upstream design docs link to source files via ../../<repo path>.
        # Redirect those to GitHub so they resolve from the published site.
        markdown = _DESIGN_REPO_PATH.sub(rf"{UPSTREAM_BLOB}/\1", markdown)

    notice = _fallback_notice(page)
    if notice:
        # Below the H1, so the page still opens with its own title.
        lines = markdown.split("\n")
        after = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), -1) + 1
        lines.insert(after, f"\n{notice}")
        markdown = "\n".join(lines)

    return markdown


# Benchmarks pages in nav order — `DATA_PAGES` in the renderer, repeated here
# because the nav is built before the renderer has run. A page the renderer did
# not produce is left out; one it produced that is not listed here is appended.
_BENCH_ORDER = [
    "index.md", "reading.md",
    "elementwise.md", "rope.md", "reduction.md", "normalization.md",
    "conv-pool.md", "gemm.md", "quantization.md", "attention.md", "moe.md",
    "sampling.md", "linear-attention.md", "ssm.md", "other.md",
]


def _bust_css_cache(config):
    """Append `?v=<content hash>` to each local stylesheet in `extra_css`."""
    out = []
    for entry in config["extra_css"]:
        path = os.path.join(config["docs_dir"], str(entry))
        if "?" in str(entry) or not os.path.isfile(path):
            out.append(entry)
            continue
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:10]
        out.append(f"{entry}?v={digest}")
    config["extra_css"] = out


def on_config(config):
    """Version the stylesheet URL; expand the Benchmarks nav entry."""
    _bust_css_cache(config)
    bench_dir = os.path.join(config["docs_dir"], "benchmarks")
    if not os.path.isdir(bench_dir):
        return config
    present = {f for f in os.listdir(bench_dir) if f.endswith(".md")}
    ordered = [f for f in _BENCH_ORDER if f in present]
    ordered += sorted(present - set(ordered))
    entries = [f"benchmarks/{f}" for f in ordered]

    for section in config["nav"]:
        if isinstance(section, dict) and "Benchmarks" in section:
            section["Benchmarks"] = entries
    return config


_FIRST_PARAGRAPH = re.compile(r"(</h1>\s*)<p>")

# Section indexes whose page lists render as cards. Each item is a link followed
# by a one-line description on the next line.
_CARD_INDEXES = {
    "user-guide/index.md", "user-guide/index.zh.md", "design/index.md",
}


def on_page_content(html, page, config, files):
    """Mark a blog post's subtitle, and the page lists on a section index."""
    src = page.file.src_path.replace("\\", "/")
    name = src.split("/")[-1]
    if src in _CARD_INDEXES:
        return html.replace("<ul>", '<ul class="page-cards">')
    if not src.startswith("blog/") or name.startswith("index."):
        return html
    return _FIRST_PARAGRAPH.sub(r'\1<p class="post-subtitle">', html, count=1)


# Old page URL -> where its content lives now, relative to the old URL. The
# redirect is written into each locale's build.
_REDIRECTS = {
    "api/topk/": "../sampling/#top-k-selection",
}

_REDIRECT_PAGE = (
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta http-equiv="refresh" content="0; url={to}">'
    '<link rel="canonical" href="{to}"><title>Redirecting</title></head>'
    '<body><a href="{to}">{to}</a></body></html>\n'
)


def _locale_dirs(config):
    """The site root, then one subdirectory per other locale i18n builds.

    This hook runs before i18n builds the other locales, so their directories
    are named from the plugin's config rather than found on disk.
    """
    dirs = [""]
    i18n = config["plugins"].get("i18n")
    for lang in getattr(i18n, "config", {}).get("languages", []):
        if lang.build and not lang.default:
            dirs.append(lang.locale)
    return dirs


def on_post_build(config):
    """Write a redirect page at each old URL `_REDIRECTS` lists, per locale."""
    for locale in _locale_dirs(config):
        for old, to in _REDIRECTS.items():
            path = os.path.join(config["site_dir"], locale, old, "index.html")
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_REDIRECT_PAGE.format(to=to))
