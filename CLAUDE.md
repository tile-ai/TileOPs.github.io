# CLAUDE.md

## Project Overview

TileOPs.github.io is the official documentation site for [TileOPs](https://github.com/tile-ai/TileOPs) — a high-performance LLM operator library built on TileLang. Deployed via GitHub Pages with GitHub Actions (`gh-pages` branch).

## Tech Stack

- **Static site generator**: [MkDocs](https://www.mkdocs.org/) with [Material for MkDocs](https://squidfundamentals.github.io/mkdocs-material/)
- **Theme**: Material — slate (dark) scheme, teal/cyan accent, Nunito + Source Code Pro fonts
- **Deployment**: GitHub Actions → `gh-pages` branch (`.github/workflows/deploy.yml`)
- **License**: MIT (tile-ai)

## Related Repositories

| Repo | Purpose |
|------|---------|
| `TileOPs` | Main operator library (source code) |

## Site Structure

```
mkdocs.yml                    # MkDocs configuration (nav, theme, extensions)
.github/workflows/deploy.yml  # Auto-deploy on push to main
docs/
  index.md                    # Home
  design/                     # Architecture & design docs
  api/                        # Operator API reference
  benchmarks/                 # Performance benchmarks
  performance-guides/          # Performance optimization guides
```

## Development

```bash
pip install mkdocs-material
mkdocs serve
```

## Build Artifacts (gitignored)

`site/`, `__pycache__/`, `.cache/`

## Collaboration Rules for Claude

- This is a documentation repo — clarity and accuracy are top priorities.
- Keep page structure consistent across the site.
- When adding new pages, update `nav` in `mkdocs.yml` accordingly.
- Use MkDocs admonitions (`!!! note`, `!!! warning`, etc.) for callouts.
- Prefer minimal, targeted changes; avoid unrelated reformatting.
- Use relative Markdown links for internal cross-references.
- Do not duplicate content that belongs in the main TileOPs repo docs; link to it instead.
- Benchmark pages have placeholder tables — they will be filled by CI automation.
- Response should include: change summary, affected paths, and next suggestions.
