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
# design pages mirror its docs/design. A symlink to a local clone works —
# ./TileOPs is gitignored.
ln -s ../TileOPs TileOPs

mkdocs serve
```

`mkdocs serve` prints the URL to open; it carries the `site_url` subpath, so the
page is at `/TileOPs.github.io/`, not `/`. Without a TileOPs checkout, mkdocstrings
cannot import `tileops` and the build aborts.

`bash scripts/render_bench.sh` fetches the nightly benchmark snapshot and renders
`docs/benchmarks/`. Those pages are generated, never edited by hand.

## Layout

```
mkdocs.yml                 # nav, theme, plugins
hooks.py                   # path rewrites for mirrored content; Benchmarks nav
scripts/                   # the Benchmarks renderer and its fetch script
docs/index.md              # Home
docs/torch-compile.md      # bringing an op into torch.compile
docs/backends.md           # adding a hardware backend
docs/api/                  # generated from docstrings by mkdocstrings
docs/benchmarks/           # generated at deploy time; only index.md is tracked
docs/performance-guides/   # trace guide, and where to find which numbers
docs/design/               # mirrored from TileOPs via include-markdown
docs/assets/extra.css      # palette and type, shared with TileFoundry
```

## Bilingual pages

English lives at the site root, Chinese under `/zh/`. A Chinese page is a
`<name>.zh.md` beside the English `<name>.md`, written as prose rather than as an
include shell — this repo holds the Chinese source of truth, and for
`torch-compile` and `backends` the `.zh.md` is edited first, with the English page
brought in line afterwards. A page with no translation falls back to English at the
same URL, with a notice prepended by `hooks.py`.

`CLAUDE.md` carries the conventions in full: the Benchmarks rules, the Chinese
typography rules, and what belongs here rather than in the TileOPs repo.
