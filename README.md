# TileOPs Documentation

The documentation site for [TileOPs](https://github.com/tile-ai/TileOPs) — spec-driven
LLM operators across backends, built on TileLang. MkDocs + Material, deployed to
`gh-pages` by GitHub Actions on every push to `main`.

**Site**: [tile-ai.github.io/TileOPs.github.io](https://tile-ai.github.io/TileOPs.github.io/)

## Local development

```bash
pip install mkdocs-material "mkdocstrings[python]" mkdocs-include-markdown-plugin \
  mkdocs-static-i18n jieba pyyaml

# The API pages read docstrings from a TileOPs checkout at ./TileOPs, and the
# design pages mirror its docs/design. The workflows clone it there; locally,
# clone it too, or point a symlink at a clone you already have. Either way the
# path is gitignored.
git clone --depth 1 https://github.com/tile-ai/TileOPs.git TileOPs

mkdocs serve
```

`mkdocs serve` prints the URL to open; it carries the `site_url` subpath, so the
page is at `/TileOPs.github.io/`, not `/`. Without a TileOPs checkout, mkdocstrings
cannot import `tileops` and the build aborts.

`bash scripts/render_bench.sh` fetches the nightly benchmark snapshot and renders
`docs/benchmarks/`. Those pages are generated, never edited by hand.

## Layout

Pages live under `docs/`, and `mkdocs.yml` holds the nav, theme and plugins — the
nav is the current page list. Three groups of pages are not written by hand:
`docs/api/` is generated from TileOPs docstrings by mkdocstrings, `docs/design/`
mirrors that repo's `docs/design/` at build time, and `docs/benchmarks/` is
rendered from the nightly snapshot at deploy time. `hooks.py` rewrites the paths
mirrored content brings with it and expands the Benchmarks nav.

## Bilingual pages

English lives at the site root, Chinese under `/zh/`. A Chinese page is a
`<name>.zh.md` beside the English `<name>.md`, written as prose rather than as an
include shell — this repo holds the Chinese source of truth, and for
`torch-compile` and `backends` the `.zh.md` is edited first, with the English page
brought in line afterwards. A page with no translation falls back to English at the
same URL, with a notice prepended by `hooks.py`.

`CLAUDE.md` carries the conventions in full: the Benchmarks rules, the Chinese
typography rules, and what belongs here rather than in the TileOPs repo.
