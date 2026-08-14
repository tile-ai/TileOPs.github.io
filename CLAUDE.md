# CLAUDE.md

## Project Overview

TileOPs.github.io is the official documentation site for [TileOPs](https://github.com/tile-ai/TileOPs) — spec-driven LLM operators across backends, built on TileLang. Deployed via GitHub Pages with GitHub Actions (`gh-pages` branch).

## Tech Stack

- **Static site generator**: [MkDocs](https://www.mkdocs.org/) with [Material for MkDocs](https://squidfundamentals.github.io/mkdocs-material/)
- **Theme**: Material, one light scheme. Palette and type are shared with [TileFoundry](https://github.com/tile-ai/TileFoundry.github.io): warm paper ground, one red accent, Bricolage Grotesque / Hanken Grotesk / Space Mono, in `docs/assets/extra.css`
- **Deployment**: GitHub Actions → `gh-pages` branch (`.github/workflows/deploy.yml`)
- **License**: MIT (tile-ai)

## Related Repositories

| Repo | Purpose |
|------|---------|
| `TileOPs` | Main operator library (source code) |

## Site Structure

```
mkdocs.yml                    # MkDocs configuration (nav, theme, extensions)
hooks.py                      # Path rewrites for mirrored content; Benchmarks nav
scripts/
  render_bench.sh             # Fetches the nightly-bench snapshot, calls the renderer
  gen_bench_pages.py          # Writes the Benchmarks pages from that snapshot
.github/workflows/
  deploy.yml                  # Auto-deploy on push to main
  render-benchmarks.yml       # Daily re-render from the latest snapshot
docs/
  index.md                    # Home
  assets/extra.css            # Palette and data-table styling
  design/                     # Architecture & design docs
  api/                        # Operator API reference
  benchmarks/                 # Generated at deploy time; only index.md is tracked
  performance-guides/         # Performance optimization guides
```

## Development

```bash
pip install mkdocs-material "mkdocstrings[python]" mkdocs-include-markdown-plugin pyyaml
mkdocs serve
```

The API reference and the hardware ceilings on the Benchmarks page read from a
TileOPs checkout at `./TileOPs` (the deploy workflow clones it there). To render
the Benchmarks pages locally, run `bash scripts/render_bench.sh` first; it
overwrites `docs/benchmarks/`, which is a build artifact.

## Build Artifacts (gitignored)

`site/`, `__pycache__/`, `.cache/`

## Collaboration Rules for Claude

- This is a documentation repo — clarity and accuracy are top priorities.
- Keep page structure consistent across the site.
- When adding new pages, update `nav` in `mkdocs.yml` accordingly. The Benchmarks section is the exception: `hooks.py` expands that one entry to whichever pages the renderer produced.
- Use MkDocs admonitions (`!!! note`, `!!! warning`, etc.) for callouts.
- Prefer minimal, targeted changes; avoid unrelated reformatting.
- Use relative Markdown links for internal cross-references.
- Do not duplicate content that belongs in the main TileOPs repo docs; link to it instead.
- Benchmark pages are generated from the nightly snapshot at deploy time, never edited by hand. `docs/benchmarks/index.md` is a committed placeholder for the case where no snapshot exists; the rest are gitignored.
- Response should include: change summary, affected paths, and next suggestions.
