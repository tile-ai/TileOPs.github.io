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
* Home opens with a card for the newest blog post, which `on_page_content`
  writes from the first entry on the Blog page.
"""
from __future__ import annotations

import hashlib
import html as htmllib
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


_POST_LINK = re.compile(r"\]\(([\w-]+)\.md\)")

_HOME_POST_LABELS = {
    "en": ("From the blog", "Read the post"),
    "zh": ("博客", "阅读全文"),
}


def _read_md(docs_dir, stem, locale):
    """The `<stem>.<locale>.md` source, else `<stem>.md`."""
    for name in (f"{stem}.{locale}.md", f"{stem}.md"):
        path = os.path.join(docs_dir, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
    return ""


def _front_matter(lines):
    """`key: value` pairs of a page's YAML front matter; flat values only."""
    if not lines or lines[0].strip() != "---":
        return {}
    meta = {}
    for ln in lines[1:]:
        if ln.strip() == "---":
            break
        key, sep, value = ln.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta


def _coming_soon_slug(docs_dir):
    """The slug of a post whose front matter says `status: coming-soon`, or None."""
    blog = os.path.join(docs_dir, "blog")
    if not os.path.isdir(blog):
        return None
    for name in sorted(os.listdir(blog)):
        stem = name.split(".")[0]
        if not name.endswith(".md") or stem == "index":
            continue
        with open(os.path.join(blog, name), encoding="utf-8") as f:
            meta = _front_matter(f.read().splitlines())
        if meta.get("status") == "coming-soon":
            return stem
    return None


def _home_post_card(docs_dir, locale):
    """Card HTML for the first post the Blog page lists, else for a post marked
    `status: coming-soon`, or "" when there is neither."""
    link = _POST_LINK.search(_read_md(docs_dir, "blog/index", locale))
    slug = link.group(1) if link else _coming_soon_slug(docs_dir)
    if not slug:
        return ""
    lines = _read_md(docs_dir, f"blog/{slug}", locale).splitlines()
    meta = _front_matter(lines)
    h1 = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), None)
    if h1 is None:
        return ""
    title = lines[h1][2:].strip()
    subtitle = next((ln.strip() for ln in lines[h1 + 1:] if ln.strip()), "")
    kicker, more = _HOME_POST_LABELS.get(locale, _HOME_POST_LABELS["en"])
    kicker = meta.get("kicker", kicker)
    esc = htmllib.escape
    desc = meta.get("description", "")
    soon = meta.get("status") == "coming-soon"
    if soon:
        # Not published yet: the card announces the post and links nowhere.
        kicker = more = "Coming soon"
        open_tag, close_tag = '<div class="home-post home-post--soon">', "</div>"
    else:
        open_tag, close_tag = f'<a class="home-post" href="blog/{slug}/">', "</a>"
    return (
        open_tag
        + f'<span class="home-post__kicker">{esc(kicker)}</span>'
        f'<span class="home-post__title">{esc(title)}</span>'
        f'<span class="home-post__subtitle">{esc(subtitle)}</span>'
        + (f'<span class="home-post__desc">{esc(desc)}</span>' if desc else "")
        + f'<span class="home-post__more">{esc(more)}</span>'
        + close_tag
    )


def on_page_content(html, page, config, files):
    """Mark a blog post's subtitle and a section index's page lists; put the
    newest post's card on Home."""
    src = page.file.src_path.replace("\\", "/")
    name = src.split("/")[-1]
    if src in _CARD_INDEXES:
        return html.replace("<ul>", '<ul class="page-cards">')
    if "/" not in src and name.startswith("index."):
        locale = getattr(page.file, "locale", None) or "en"
        card = _home_post_card(config["docs_dir"], locale)
        return html.replace("</h1>", "</h1>\n" + card, 1) if card else html
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
