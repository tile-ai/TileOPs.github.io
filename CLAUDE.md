# CLAUDE.md

MkDocs + Material docs site for [TileOPs](https://github.com/tile-ai/TileOPs),
deployed to `gh-pages` by GitHub Actions. Served at `/TileOPs.github.io/`, the
subpath `site_url` carries. Palette and type come from
[TileFoundry](https://github.com/tile-ai/TileFoundry.github.io) —
`docs/assets/extra.css`.

## Development

```bash
bash scripts/dev.sh serve     # or: build, bench
```

- Sets up a `.venv` from `requirements-docs.txt` and the `./TileOPs` checkout.
  `build` applies `checks.yml`'s warning rule; `bench` calls `render_bench.sh`.
- Both are reused and stay at the revision they were cloned at. Pass `--update`
  before trusting a local build against upstream.
- The checkout is required: `api/` reads its docstrings, `design/` mirrors its
  `docs/design/`. Without it the build aborts.

## Checks

`.github/workflows/checks.yml`, on every push and pull request:

| Command | Holds |
|---------|-------|
| `pytest` | The renderer, and `tests/golden/` byte for byte |
| `ruff check scripts hooks.py tests` | `pyproject.toml`. `E501` off — wrap prose by hand |
| `npx stylelint "docs/assets/**/*.css"` | No duplicate selector, no `color-mix()`. Why: an engine that cannot parse a function drops the whole declaration |
| `mkdocs build` | Any warning of ours fails. griffe's come from TileOPs' docstrings and do not gate this repo |
| `python scripts/check_api_pages.py` | Every `::: tileops.<family>.<Op>` under `docs/api/` is in that family's `__all__`. An exported op no page names is printed, not failed |

- Keep the suite small; `pytest --collect-only -q` lists it.
- The parametric-workload test needs a checkout (`./TileOPs`, or `$TILEOPS`) and
  skips without one. CI's Renderer job provides it.
- `tests/fixtures/` is a trimmed snapshot, `tests/golden/` the three data pages
  it must produce. A change to what they say fails by design: read the diff,
  then run `python tests/refresh_golden.py`.
- Add a unit test only for a rule the golden pages do not show.

## Generated pages

Never edit by hand — change what produces them.

| Pages | Produced by |
|-------|-------------|
| `docs/api/` | mkdocstrings, from TileOPs docstrings |
| `docs/design/` | `include-markdown`, mirroring TileOPs `docs/design/` |
| `docs/benchmarks/` | `scripts/gen_bench_pages.py`, from the newest commit on the `snapshots` branch of [tile-ai/TileOPs-nightly](https://github.com/tile-ai/TileOPs-nightly) (`scripts/render_bench.sh` fetches it) |

`hooks.py` rewrites the repo-relative paths mirrored content arrives with, and
expands the single `Benchmarks` nav entry to the pages the renderer produced.

Which ops a `docs/api/` page names is written by hand and drifts as TileOPs adds
and removes ops. The deploy and the daily refresh run
`scripts/check_api_pages.py` against the checkout they just made.

## Benchmarks pages

One question per workload: how TileOPs compares to the fastest other
implementation of the same op on that workload.

| Rule | Detail |
|------|--------|
| No aggregates | One table per op, one row per workload. Why: an op's workloads span shapes orders of magnitude apart, so a median matches no reproducible run |
| The colour is the verdict | `Ratio` sits right after the workload name: red behind, plain ink level, green ahead, grey where the only rival is an eager `-ref` |
| Device time | Compare `device_busy_ms`, never wall-clock span |
| Two questions | `Ratio`: is another kernel faster. `SOL`: how much faster the hardware allows anyone to go, its binding resource (`mem`/`comp`/`lat`) in a `Bound` column. Import the SOL arithmetic and thresholds from the checkout's roofline tool (M5); never re-derive them here |
| Which page an op lands on | The manifest entry's `family:`, through `_MANIFEST_FAMILY` — an op TileOPs adds needs no change here. One the manifest does not declare falls back to its package, then to keywords |
| Page order | `DATA_PAGES`: Elementwise, RoPE, Reduction, Normalization, Conv & Pool, GEMM, Quantization & Dequantization, Attention, MoE, Sampling, Linear Attention, SSM, Other. One page per family except `Conv & Pool` (two) and `Other` (FFT, mHC, Engram, the rest). `TopkSelectorFwdOp` declares `family: attention`, so its row is on Attention, while the API Reference documents it on the Sampling page. The API Reference nav follows it, with FFT, mHC and Engram after SSM and Top-k on the Sampling page. `_BENCH_ORDER` in `hooks.py` repeats it — change all three together |
| Op order within a page | The order `docs/api/` names them, read by `api_op_order()`. An op no API page names comes last, ranked by verdict |
| Rows follow the manifest | One row group per manifest label, one row per dtype under it in a `dtype` column. Labels keep the snapshot's order, which is the manifest's; the key above the table repeats it. A row no manifest describes takes its id, trailing dtype names split off, as its label |
| Workload shapes | The snapshot names a workload but carries no shapes. `scripts/workload_shape.py` reads them from the spec manifest at the commit the benchmark ran on, joined by the `<label>-<dtype>` the benchmark id is built from. A workload the manifest does not declare keeps its id and gets no shapes — never a guessed one |

## Bilingual pages (en / zh)

English at the site root, Chinese under `/zh/`. A Chinese page is a
`<name>.zh.md` beside the English `<name>.md`, full prose, never an
`include-markdown` shell. `backends.md`, `torch-compile.md`, everything under
`performance-guides/memory-bound/`, `blog/`, `user-guide/development.md` and the
two guides under `user-guide/manifest/` and `user-guide/dispatch/` were authored
in Chinese: edit the `.zh.md` first, then bring the English page in line.
Everything else goes the other way.

| Rule | Detail |
|------|--------|
| Coverage | Whichever pages have a `.zh.md` — `ls docs/**/*.zh.md` |
| Never translate | `api/` and `benchmarks/`, both generated; `design/`, mirrored English |
| Mirrored, translated | `performance-guides/trace-timeline.md` mirrors TileOPs `docs/perf/trace-timeline.md`; its `.zh.md` is a full translation. When upstream changes that file, update the translation in step |
| Missing translation | Falls back to English at the same URL, and `hooks.py` prepends a "本页暂无中文版" notice. The fallback runs zh → en only: a page that exists only as `.zh.md` leaves its `nav` entry on a missing file and the English sidebar renders a dead link |
| Figures | A figure with text needs one SVG per language: `img/<name>.svg` for English and `img/<name>.zh.svg` beside it, which the `zh` build picks up for the same reference. Translate the `<text>` nodes and the `aria-label`, keep the geometry. English runs longer than Chinese — grow the `viewBox` rather than let text overflow. The user-guide figures are drawn from sources under `figures/user-guide/` (`<name>.zh.puml`, `<name>.en.puml`, `manifest/overview.py`): edit the source, then run `figures/user-guide/render.sh` |
| Nav labels | `nav_translations` in the `i18n` plugin block; keep an entry for every `nav` title |
| Chinese search | Requires `jieba` |
| Punctuation | Full-width in Chinese prose: `，。：；（）`. Latin quotes and brackets stay half-width inside code spans |
| Latin in Chinese | A space either side of a Latin token: `由 spec 驱动`, `形状和 dtype`. Not inside code spans |
| Keep in English | kernel, spec, agent, dtype, roofline, GEMM, target, op, family, backend, and every op name. Why: translating them loses the link to the API |
| Terms | 「kernel 接口」 for kernel interface, 「in-tree 实现」 for an in-tree implementation, 「build identity」 and 「构建函数」 for what `entry_for` returns |
| Inline code | Real identifiers only (`GemmOp`, `eval_roofline`, paths, flags). A concept mentioned in prose is not code |
| Type | `extra.css` gives `html[lang="zh"]` looser leading and headings at 700, not 800 — at 800 a CJK fallback face closes up the strokes. Scoped away from fallback pages, whose body text is English. No CJK webfont: Han glyphs come from the platform UI face (`--tf-cjk`) |

## Nav

`nav` in `mkdocs.yml` is the page list: Home, then Blog, then the remaining
sections in the reader's order, Design last as contributor-facing.

- Add a new page to `nav`, and its label to `nav_translations`.
- Put a user-facing topic under User Guide, in the group its index lists it in,
  and keep the nav in the index's order. The index is a plain list per group;
  `hooks.py` marks it and `extra.css` draws each item as a card.
- Keep a label short enough for one line in the sidebar; the page's H1 carries
  the full title.
- Merging a page into another: delete it, and add its old URL to `_REDIRECTS`
  in `hooks.py` so published links redirect.
- List a section's `<dir>/index.md` as a bare path with no title. Given a title
  it is promoted anyway and its sidebar row disappears.
- Leave `toc.integrate` off: the page TOC renders in the right column, and it is
  incompatible with `navigation.indexes`.

## Blog

`docs/blog/`: plain pages, not Material's `blog` plugin. Why: the plugin
renders no post under `mkdocs-static-i18n` and warns on its archive pages.

- One post is `blog/<slug>.zh.md` and `blog/<slug>.md`, listed in `nav` under
  Blog and as one link on `blog/index.md`, newest first.
- A post opens with a short H1 and a one-line subtitle paragraph. `hooks.py`
  gives that paragraph the `post-subtitle` class that `extra.css` styles; keep
  classes and attribute lists out of the post's Markdown.
- No date line, no in-page TOC: the right column carries the TOC.

## Conventions

- Measure every number and state its conditions. Say when a count will drift.
- Admonitions (`!!! note`, `!!! warning`) for callouts; relative Markdown links
  for internal cross-references.
- Write links as plain Markdown. `extra.css` gives every link in running text an
  arrow, east within the site and north-east off it, and a teal wash on hover.
- Link to the TileOPs repo rather than duplicating it. A page authored here that
  mirrors upstream content will drift.
- Gitignored: `site/`, `__pycache__/`, `.cache/`, `TileOPs/`, and
  `docs/benchmarks/` except `index.md`.
