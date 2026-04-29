# Catalog

The full per-skill reference, mirrored from
[`docs/tileops-skills.md`](https://github.com/tile-ai/TileOPs/blob/main/docs/tileops-skills.md)
in `tile-ai/TileOPs`. See the [Overview](index.md) for the at-a-glance
matrix, intent table, and trust model.

## How skills are organized

TileOPs skills are organized along **two axes**:

**Scope** — what the skill operates on:

- **op** — a single operator.
- **family** — a group of related ops (reductions, norms, …).
- **manifest** — entries in `tileops/manifest/`.

**Role** — how the skill relates to others:

- **orchestrator** — calls other skills; the day-to-day entry point.
- **atomic** — does one job; usually invoked by an orchestrator
  (standalone use is for debugging).

Combining the two yields five groups, which are how the rest of this
page is laid out:

| Scope ↓ &nbsp;/&nbsp; Role → | Orchestrator | Atomic |
|:--|:--|:--|
| **family**   | `align-family` | `audit-family` |
| **op**       | `align-op`     | `scaffold-op`, `implement-op`, `test-op`, `bench-op` |
| **manifest** | —              | `add-manifest`, `fix-manifest` |

Manifest skills have no orchestrator — they precede op-layer work,
not contain it. The diagram below makes the delegation concrete.

## Composition

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="## Composition"
   end="## Trust model"
%}

## Skills in detail

### Per-family orchestrator

#### `align-family`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `align-family`"
   end="### `scaffold-op`"
%}

### Per-family atomic

#### `audit-family`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `audit-family`"
   end="### `add-manifest`"
%}

### Per-op orchestrator

#### `align-op`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `align-op`"
   end="### `align-family`"
%}

### Per-op atomic

#### `scaffold-op`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `scaffold-op`"
   end="### `implement-op`"
%}

#### `implement-op`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `implement-op`"
   end="### `test-op`"
%}

#### `test-op`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `test-op`"
   end="### `bench-op`"
%}

#### `bench-op`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `bench-op`"
   end="### `audit-family`"
%}

### Manifest atomic

#### `add-manifest`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `add-manifest`"
   end="### `fix-manifest`"
%}

#### `fix-manifest`

{%
   include-markdown "../../TileOPs/docs/tileops-skills.md"
   start="### `fix-manifest`"
   end="## Composition"
%}
