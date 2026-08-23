# CLAUDE.md

Documentation site for [TileOPs](https://github.com/tile-ai/TileOPs) — spec-driven
LLM operators across backends, built on TileLang. MkDocs + Material, deployed to
`gh-pages` by GitHub Actions. Palette and type are shared with
[TileFoundry](https://github.com/tile-ai/TileFoundry.github.io); see
`docs/assets/extra.css`.

## Development

```bash
pip install mkdocs-material "mkdocstrings[python]" mkdocs-include-markdown-plugin \
  mkdocs-static-i18n jieba pyyaml
git clone --depth 1 https://github.com/tile-ai/TileOPs.git TileOPs   # or symlink one
mkdocs serve
```

The checkout at `./TileOPs` is required, not optional: `api/` reads its docstrings
and `design/` mirrors its `docs/design/`. Without it, mkdocstrings cannot import
`tileops` and the build aborts — with or without `--strict`.

`mkdocs serve` serves under the `site_url` subpath, so the page is at
`/TileOPs.github.io/`. `bash scripts/render_bench.sh` renders `docs/benchmarks/`.

## Generated pages

Never edit these by hand — change what produces them.

| Pages | Produced by |
|-------|-------------|
| `docs/api/` | mkdocstrings, from TileOPs docstrings |
| `docs/design/` | `include-markdown`, mirroring TileOPs `docs/design/` |
| `docs/benchmarks/` | `scripts/gen_bench_pages.py`, from the nightly snapshot (`scripts/render_bench.sh` fetches it) |

`hooks.py` rewrites the repo-relative paths mirrored content arrives with, and
expands the single `Benchmarks` nav entry to whichever pages the renderer produced.

## Benchmarks pages

They answer one question per workload: how TileOPs compares to the fastest other
implementation of the same op on that workload.

| Rule | Detail |
|------|--------|
| No aggregates | One table per op, one row per workload. An op's workloads span shapes orders of magnitude apart, so a median matches no reproducible run and a mean ratio hides which shape is behind. |
| The colour is the verdict | `Ratio` sits right after the workload name: red behind, plain ink level, green ahead, grey where the only rival is an eager `-ref`. |
| Device time | The compared quantity is `device_busy_ms`, never wall-clock span. |
| Not reported | Utilisation against a hardware ceiling (SOL, bound, arithmetic intensity) — a different question. |

## Bilingual pages (en / zh)

English lives at the site root, Chinese under `/zh/`. A Chinese page is a
`<name>.zh.md` beside the English `<name>.md` — full prose, never an
`include-markdown` shell.

`backends.md` and `torch-compile.md` were authored in Chinese: edit the `.zh.md`
first, then bring the English page in line. Everything else goes the other way.

| Rule | Detail |
|------|--------|
| Coverage | `index.md`, `backends.md`, `torch-compile.md`, `performance-guides/index.md`. |
| Never translate | `api/` and `benchmarks/` — both generated. `design/` is mirrored English. |
| Missing translation | Falls back to English at the same URL, so the zh nav is never sparse. `hooks.py` prepends a "本页暂无中文版" notice. |
| Nav labels | `nav_translations` in the `i18n` plugin block; keep an entry for every `nav` title. |
| Chinese search | Requires `jieba`. |

### Chinese typography

| Rule | Detail |
|------|--------|
| Punctuation | Full-width in Chinese prose: `，。：；（）`. Latin quotes and brackets stay half-width inside code spans. |
| Latin in Chinese | A space either side of a Latin token: `由 spec 驱动`, `形状和 dtype`. Not inside code spans. |
| Keep in English | kernel, spec, agent, dtype, roofline, GEMM, target, and every op name — translating them loses the link to the API. |
| Inline code | Real identifiers only (`GemmOp`, `eval_roofline`, paths, flags). A concept mentioned in prose is not code. |
| Type metrics | `extra.css` sets looser leading under `html[lang="zh"]` and drops headings from 800 to 700 — a Latin display face at 800 falls through to a CJK face where that weight closes up the strokes. The body size is the Latin one. Scoped away from fallback pages, whose body text is English. |
| No CJK webfont | Han glyphs come from the platform UI face (`--tf-cjk`); a Simplified Chinese subset costs megabytes per page load. |

## Nav

Six sections, ordered by the reader's questions — what it is, how to use it, what
to call, how fast it is, how to make it faster, how it works inside: Home, User
Guide, API Reference, Benchmarks, Performance Guides, Design. Design comes last;
it is contributor-facing and mirrored from TileOPs.

- Add a new page to `nav` in `mkdocs.yml`, and its label to `nav_translations`.
- A user-facing topic goes under User Guide.
- Keep a label short enough to sit on one line in the sidebar; the page's own H1
  carries the full title.
- `<dir>/index.md` is a section's own overview page — `navigation.indexes` promotes
  it, listed in `nav` as a bare path with no title. Any other page keeps its own
  name: an `index.md` given a title in `nav` is promoted anyway, and its row
  disappears from the sidebar.
- The page TOC renders in the right column, so `toc.integrate` stays off — it is
  incompatible with `navigation.indexes`.

## Conventions

- Clarity and accuracy come first — this is documentation.
- Every number is measured, with its conditions stated. Say when a count will drift.
- Admonitions (`!!! note`, `!!! warning`) for callouts; relative Markdown links for
  internal cross-references.
- Minimal, targeted changes; no unrelated reformatting.
- Don't duplicate what belongs in the TileOPs repo — link to it. A page authored
  here that mirrors upstream content will drift.
- Gitignored: `site/`, `__pycache__/`, `.cache/`, `TileOPs/`, and `docs/benchmarks/`
  except `index.md`.
