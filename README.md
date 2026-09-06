# TileOPs Documentation

The documentation site for [TileOPs](https://github.com/tile-ai/TileOPs) — spec-driven
LLM operators across backends, built on TileLang.

**Live site**: [tile-ai.github.io/TileOPs.github.io](https://tile-ai.github.io/TileOPs.github.io/)

MkDocs + Material. GitHub Actions deploys to `gh-pages` on every push to `main`,
and re-renders the Benchmarks pages nightly.

## Local development

```bash
bash scripts/dev.sh serve     # or: build, bench
```

It creates `.venv`, installs `requirements-docs.txt` into it, clones the TileOPs
checkout the pages read, and serves. Later runs reuse both: reinstalling only
when `requirements-docs.txt` changed, and keeping the checkout at the revision
it was cloned at. CI checks out upstream fresh every run, so `--update` before
trusting a local build is the difference. `--no-venv` uses the interpreter
already on `PATH`, `--port` moves the server.

By hand: `pip install -r requirements-docs.txt`, clone TileOPs to `./TileOPs`
(a symlink to a clone you already have works, and the path is gitignored), then
`mkdocs serve`.

Two things to know before the first run.

- **The URL carries a subpath.** `site_url` ends in `/TileOPs.github.io/`, so
  that, not `/`, is the page `mkdocs serve` prints.
- **The checkout is not optional.** Without it, mkdocstrings cannot import
  `tileops`, and the build aborts rather than warns.

`bash scripts/dev.sh bench` puts real numbers on the Benchmarks pages. They are
generated output — change the renderer, never the pages.

## Layout

Pages live under `docs/`. `mkdocs.yml` holds the nav, theme and plugins, and its
nav is the current page list.

Three groups of pages are not written by hand:

- `docs/api/` — generated from TileOPs docstrings by mkdocstrings.
- `docs/design/` — mirrored from that repo's `docs/design/` at build time.
- `docs/benchmarks/` — rendered from the nightly snapshot at deploy time.

`hooks.py` cleans up after the first two and stands in for the third: it rewrites
the repo-relative paths mirrored content arrives with, and expands the single
Benchmarks nav entry, since those pages do not exist until the renderer has run.

## Bilingual pages

English lives at the site root, Chinese under `/zh/`. A Chinese page is a
`<name>.zh.md` beside the English `<name>.md`, written as prose rather than as an
include shell. Some pages are drafted in Chinese and translated into English, so
a page's two versions are kept in step by hand.

A page with no translation falls back to English at the same URL, with a notice
prepended by `hooks.py` — so the Chinese nav is never sparse.
