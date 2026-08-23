# CLAUDE.md

Documentation site for [TileOPs](https://github.com/tile-ai/TileOPs) — spec-driven
LLM operators across backends, built on TileLang. MkDocs + Material, deployed to
`gh-pages` by GitHub Actions. Palette and type are shared with
[TileFoundry](https://github.com/tile-ai/TileFoundry.github.io); see
`docs/assets/extra.css`.

## Layout

```
mkdocs.yml                    # nav, theme, plugins
hooks.py                      # path rewrites for mirrored content; Benchmarks nav
scripts/render_bench.sh       # fetches the nightly-bench snapshot, calls the renderer
scripts/gen_bench_pages.py    # writes the Benchmarks pages from that snapshot
.github/workflows/            # deploy.yml (push to main), render-benchmarks.yml (daily)
docs/design/                  # mirrored from TileOPs via include-markdown
docs/api/                     # generated from Python docstrings by mkdocstrings
docs/benchmarks/              # generated at deploy time; only index.md is tracked
docs/backends.md              # backend-authoring guide; Chinese is the source
docs/torch-compile.md         # calling ops under torch.compile; Chinese is the source
docs/performance-guides/
```

## Development

```bash
pip install mkdocs-material "mkdocstrings[python]" mkdocs-include-markdown-plugin \
  mkdocs-static-i18n jieba pyyaml
mkdocs serve
```

`api/` and the mirrored `design/` pages read from a TileOPs checkout
at `./TileOPs` (the workflows clone it there); without it mkdocstrings warns, and
`--strict` fails. `bash scripts/render_bench.sh` renders `docs/benchmarks/`.

## Benchmarks pages

Generated from the nightly snapshot, never edited by hand — change the renderer
instead. They answer one question per workload: how TileOPs compares to the
fastest other implementation of the same op on that workload.

| Rule | Detail |
|------|--------|
| No aggregates | One table per op, one row per workload. An op's workloads span shapes orders of magnitude apart, so a median matches no reproducible run and a mean ratio hides which shape is behind. |
| The colour is the verdict | `Ratio` sits right after the workload name: red behind, plain ink level, green ahead, grey where the only rival is an eager `-ref`. |
| Device time | The compared quantity is `device_busy_ms`, never wall-clock span. |
| Not reported | Utilisation against a hardware ceiling (SOL, bound, arithmetic intensity) — a different question. |
| Nav | `hooks.py` expands the single `Benchmarks` nav entry to whichever pages the renderer produced. |

## Bilingual pages (en / zh)

English lives at the site root, Chinese under `/zh/`. A Chinese page is a
`<name>.zh.md` beside the English `<name>.md` — full prose, not an
`include-markdown` shell, because this repo holds the Chinese source of truth
while the English design docs are mirrored from TileOPs.

`backends/` runs the other way: it was authored in Chinese, and the English page
is the translation. Edit the `.zh.md` first there, then bring `<name>.md` in line.

| Rule | Detail |
|------|--------|
| Coverage | Translate `index.md`, `design/`, and `backends/`. |
| Never translate | `api/` and `benchmarks/` — both generated. |
| Missing translation | Falls back to English at the same URL, so the zh nav is never sparse. `hooks.py` prepends a "本页暂无中文版" notice. |
| Nav labels | `nav_translations` in the `i18n` plugin block; keep an entry for every `nav` title. |
| Chinese search | Requires `jieba`. |

Translated prose should read as Chinese written from scratch, not as a
word-for-word rendering of the English.

### Chinese typography

| Rule | Detail |
|------|--------|
| Punctuation | Full-width in Chinese prose: `，。：；（）`. Latin quotes and brackets stay half-width inside code spans. |
| Latin in Chinese | A space either side of a Latin token: `由 spec 驱动`, `形状和 dtype`. Not inside code spans. |
| Keep in English | kernel, spec, agent, dtype, roofline, GEMM, and every op name — translating them loses the link to the API. |
| Inline code | Real identifiers only (`GemmOp`, `eval_roofline`, paths, flags). A concept mentioned in prose is not code. |
| Type metrics | `extra.css` sets looser leading under `html[lang="zh"]` and drops headings from 800 to 700 — a Latin display face at 800 falls through to a CJK face where that weight closes up the strokes. The body size is the Latin one. Scoped away from fallback pages, whose body text is English. |
| No CJK webfont | Han glyphs come from the platform UI face (`--tf-cjk`); a Simplified Chinese subset costs megabytes per page load. |

## Conventions

- Clarity and accuracy come first — this is documentation.
- Add new pages to `nav` in `mkdocs.yml` (Benchmarks excepted, see above).
- Six sections, ordered by the reader's questions — what it is, how to use it,
  what to call, how fast it is, how to make it faster, how it works inside: Home,
  User Guide, API Reference, Benchmarks, Performance Guides, Design. A user-facing
  topic goes under User Guide as its own subsection. Design comes last: it is
  contributor-facing and mirrored from TileOPs.
- `navigation.indexes` makes a section's first nav entry — a bare `<dir>/index.md`
  with no title — the section's own page. Its H1 has to read as the nav label, and
  the label needs a `nav_translations` entry. The page TOC renders in the right
  column, so `toc.integrate` must stay off: the two features are incompatible.
- Admonitions (`!!! note`, `!!! warning`) for callouts; relative Markdown links
  for internal cross-references.
- Minimal, targeted changes; no unrelated reformatting.
- Don't duplicate what belongs in the TileOPs repo — link to it. `design/` pages
  are mirrors; a page authored here instead will drift.
- Gitignored: `site/`, `__pycache__/`, `.cache/`, `TileOPs/`, and
  `docs/benchmarks/` except `index.md`.
- Responses: change summary, affected paths, next suggestions.
