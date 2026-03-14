# CLAUDE.md

## Project Overview

TileOPs.github.io is the official documentation site for [TileOPs](https://github.com/tile-ai/TileOPs) — a high-performance LLM operator library built on TileLang. This repo is deployed via GitHub Pages.

## Tech Stack

- **Static site generator**: [MkDocs](https://www.mkdocs.org/) with [Material for MkDocs](https://squidfundamentals.github.io/mkdocs-material/)
- **Theme**: Material — slate (dark) scheme, deep orange accent
- **Markup**: Markdown
- **License**: MIT (tile-ai)

## Related Repositories

| Repo | Purpose |
|------|---------|
| `TileOPs` | Main operator library (source code) |
| `TileOPs.wiki` | Internal wiki — meeting notes, dashboards, TileLang knowledge base |

## Site Structure

```
mkdocs.yml              # MkDocs configuration
docs/
  index.md              # Home page
  api/                  # API Reference
    index.md
    attention.md
    linear-algebra.md
    normalization.md
    reduction.md
    elementwise.md
    other.md
  design/               # Design documents
    index.md
    two-layer-architecture.md
    hardware-dispatch.md
  benchmarks/           # Performance results
    index.md
    attention.md
    gemm.md
    normalization.md
    elementwise.md
  blog/
    index.md
```

## Development

```bash
pip install mkdocs-material
mkdocs serve
# Site available at http://localhost:8000
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
