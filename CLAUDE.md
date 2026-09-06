# CLAUDE.md

Documentation site for [TileOPs](https://github.com/tile-ai/TileOPs) — spec-driven
LLM operators across backends, built on TileLang. MkDocs + Material, deployed to
`gh-pages` by GitHub Actions. Palette and type are shared with
[TileFoundry](https://github.com/tile-ai/TileFoundry.github.io); see
`docs/assets/extra.css`.

## Development

```bash
bash scripts/dev.sh serve     # or: build, bench
```

- Sets up a `.venv` from `requirements-docs.txt` and the `./TileOPs` checkout,
  then serves. `build` applies `checks.yml`'s warning rule, `bench` calls
  `render_bench.sh`.
- Both are reused, and the checkout stays at the revision it was cloned at:
  `--update` before trusting a local build against upstream.
- The checkout is required: `api/` reads its docstrings, `design/` mirrors its
  `docs/design/`, and without it the build aborts rather than warns.
- The page is at `/TileOPs.github.io/` — `site_url` carries that subpath.

## Checks

`.github/workflows/checks.yml` runs these on every push and pull request:

| Command | Holds |
|---------|-------|
| `pytest` | The renderer, and `tests/golden/` byte for byte |
| `ruff check scripts hooks.py tests` | `pyproject.toml`; `E501` off, prose is wrapped by hand |
| `npx stylelint "docs/assets/**/*.css"` | No duplicate selector, no `color-mix()` — an engine that cannot parse a function drops the whole declaration |
| `mkdocs build` | Any warning of ours fails; griffe's are TileOPs' docstrings, not this repo's gate |
| `python scripts/check_api_pages.py` | Every `::: tileops.<family>.<Op>` under `docs/api/` is in that family's `__all__`; an exported op no page names is printed, not failed |

Eight tests, and that is the intended size. `tests/fixtures/` is a trimmed
snapshot, `tests/golden/` the three data pages it must produce; a change to what
they say fails by design — read the diff, then `python tests/refresh_golden.py`.
A unit test is added only for a rule the golden pages do not show.

## Generated pages

Never edit these by hand — change what produces them.

| Pages | Produced by |
|-------|-------------|
| `docs/api/` | mkdocstrings, from TileOPs docstrings |
| `docs/design/` | `include-markdown`, mirroring TileOPs `docs/design/` |
| `docs/benchmarks/` | `scripts/gen_bench_pages.py`, from the newest commit on the `snapshots` branch of [tile-ai/TileOPs-nightly](https://github.com/tile-ai/TileOPs-nightly) (`scripts/render_bench.sh` fetches it) |

`hooks.py` rewrites the repo-relative paths mirrored content arrives with, and
expands the single `Benchmarks` nav entry to whichever pages the renderer produced.

Which ops a `docs/api/` page names is written by hand, so it drifts as TileOPs
adds and removes ops — the deploy and the daily refresh run
`scripts/check_api_pages.py` against the checkout they just made.

## Benchmarks pages

They answer one question per workload: how TileOPs compares to the fastest other
implementation of the same op on that workload.

| Rule | Detail |
|------|--------|
| No aggregates | One table per op, one row per workload. An op's workloads span shapes orders of magnitude apart, so a median matches no reproducible run. |
| The colour is the verdict | `Ratio` sits right after the workload name: red behind, plain ink level, green ahead, grey where the only rival is an eager `-ref`. |
| Device time | The compared quantity is `device_busy_ms`, never wall-clock span. |
| Two questions | `Ratio` says whether someone else's kernel is faster; `SOL` how much faster the hardware allows anyone to go, with the binding resource (`mem`/`comp`/`lat`) in its own `Bound` column. The SOL arithmetic and thresholds are imported from the checkout's roofline tool (M5) — never re-derived here. |
| Order follows the API Reference | `DATA_PAGES` takes the API nav's order over the same families — a page per family except `Conv & Pool` (two) and `Other` (Top-k, FFT, mHC, Engram, the rest). Within a page, ops sit in the order `docs/api/` names them, read by `api_op_order()`; an op no API page names comes last, ranked by verdict. `_BENCH_ORDER` in `hooks.py` repeats the page order for the nav: change one, change the other. |
| Workload shapes | The snapshot names a workload but does not carry its shapes. `scripts/workload_shape.py` reads them from the TileOPs spec manifest at the commit the benchmark ran on, joined by the `<label>-<dtype>` the benchmark id is built from. A workload the manifest does not declare keeps its id and gets no shapes — never a guessed one. |

## Bilingual pages (en / zh)

English lives at the site root, Chinese under `/zh/`. A Chinese page is a
`<name>.zh.md` beside the English `<name>.md` — full prose, never an
`include-markdown` shell. `backends.md`, `torch-compile.md` and everything under
`performance-guides/memory-bound/` were authored in Chinese: edit the `.zh.md`
first, then bring the English page in line. Everything else goes the other way.

| Rule | Detail |
|------|--------|
| Coverage | Whichever pages have a `.zh.md` — `ls docs/**/*.zh.md`. |
| Never translate | `api/` and `benchmarks/` — both generated. `design/` is mirrored English. |
| Missing translation | Falls back to English at the same URL, so the zh nav is never sparse. `hooks.py` prepends a "本页暂无中文版" notice. The fallback runs zh → en only: a page that exists only as `.zh.md` leaves its `nav` entry pointing at a missing file, and the English sidebar renders a dead link. |
| Figures | A figure with text in it needs one SVG per language — translate the `<text>` nodes and the `aria-label`, keep the geometry. English runs longer than Chinese: grow the `viewBox` rather than letting text overflow. |
| Nav labels | `nav_translations` in the `i18n` plugin block; keep an entry for every `nav` title. |
| Chinese search | Requires `jieba`. |
| Punctuation | Full-width in Chinese prose: `，。：；（）`. Latin quotes and brackets stay half-width inside code spans. |
| Latin in Chinese | A space either side of a Latin token: `由 spec 驱动`, `形状和 dtype`. Not inside code spans. |
| Keep in English | kernel, spec, agent, dtype, roofline, GEMM, target, and every op name — translating them loses the link to the API. |
| Inline code | Real identifiers only (`GemmOp`, `eval_roofline`, paths, flags). A concept mentioned in prose is not code. |
| Type | `extra.css` gives `html[lang="zh"]` looser leading and headings at 700, not 800 — at 800 a CJK fallback face closes up the strokes. Scoped away from fallback pages, whose body text is English. No CJK webfont: Han glyphs come from the platform UI face (`--tf-cjk`). |

## Nav

`nav` in `mkdocs.yml` is the page list; its six sections run in the reader's
order, Design last as contributor-facing.

- Add a new page to `nav`, and its label to `nav_translations`.
- A user-facing topic goes under User Guide.
- Keep a label short enough to sit on one line in the sidebar; the page's own H1
  carries the full title.
- `<dir>/index.md` is a section's overview page: list it in `nav` as a bare path
  with no title. Given a title it is promoted anyway, and its sidebar row
  disappears.
- `toc.integrate` stays off — the page TOC renders in the right column, and it is
  incompatible with `navigation.indexes`.

## Conventions

- Every number is measured, with its conditions stated. Say when a count will drift.
- Admonitions (`!!! note`, `!!! warning`) for callouts; relative Markdown links
  for internal cross-references.
- Don't duplicate what belongs in the TileOPs repo — link to it. A page authored
  here that mirrors upstream content will drift.
- Gitignored: `site/`, `__pycache__/`, `.cache/`, `TileOPs/`, and `docs/benchmarks/`
  except `index.md`.
