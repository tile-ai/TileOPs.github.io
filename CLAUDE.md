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
pip install mkdocs-material "mkdocstrings[python]" mkdocs-include-markdown-plugin \
  mkdocs-static-i18n jieba pyyaml
mkdocs serve
```

The API reference and the mirrored design/skills pages read from a TileOPs
checkout at `./TileOPs` (the deploy workflow clones it there). To render the
Benchmarks pages locally, run `bash scripts/render_bench.sh` first; it
overwrites `docs/benchmarks/`, which is a build artifact.

The Benchmarks pages answer one question per op: how TileOPs compares to the
fastest other implementation of the same op on the same workload. The gap is the
column right after the op name and its colour is the verdict — red behind, plain
ink level, green ahead, grey where the only rival is an eager reference.
Utilisation against a hardware ceiling (SOL, bound, arithmetic intensity) is a
different question and is deliberately not reported.

## Bilingual Pages (en / zh)

English is the default language and lives at the site root; Chinese is served
under `/zh/`. A Chinese page is a `<name>.zh.md` file beside the English
`<name>.md` — full prose, not an `include-markdown` shell, because this repo
holds the Chinese source of truth while the English design docs are mirrored
from TileOPs.

| Rule | Detail |
|------|--------|
| Coverage | Translate `index.md` and `design/`. Leave `skills/` in English — those pages are agent-facing instructions. |
| Never translate | `api/` (generated from Python docstrings) and `benchmarks/` (generated data tables). |
| Missing translation | Falls back to the English page at the same URL, so the zh nav is never sparse. `hooks.py` prepends a "本页暂无中文版" notice there. |
| Nav labels | `nav_translations` in the `i18n` plugin block. Keep an entry for every new `nav` title; an untranslated title renders in English. |
| Chinese search | Requires `jieba`. Without it, Chinese search returns nothing useful. |

Translated prose should read as Chinese written from scratch, not as a
word-for-word rendering of the English.

### Chinese Typography

| Rule | Detail |
|------|--------|
| Punctuation | Full-width in Chinese prose: `，。：；（）` — never `,.:;()`. Latin quotes and brackets stay half-width inside code spans. |
| Latin in Chinese | Keep a space either side of a Latin token: `由 spec 驱动`, `形状和 dtype`. Do not add the space inside code spans. |
| Terms to keep English | kernel, spec, agent, dtype, roofline, GEMM, and every op name. Translating them loses the link to the API. |
| Inline code | For real identifiers only (`GemmOp`, `eval_roofline`, paths, flags). A concept mentioned in prose is not code. |
| Type metrics | `extra.css` sets a larger size and looser leading under `html[lang="zh"]`, and drops headings from 800 to 700 — a Latin display face at 800 falls through to a CJK face where that weight closes up the strokes. Scoped away from fallback pages, whose body text is English. |
| No CJK webfont | Han glyphs come from the platform UI face (`--tf-cjk`). A Simplified Chinese subset costs megabytes per page load. |

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
