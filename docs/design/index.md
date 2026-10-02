# Design

Architecture and design documentation for TileOPs internals. The pages
below mirror `docs/design/` in the [`tile-ai/TileOPs`](https://github.com/tile-ai/TileOPs)
repository — the source of truth — pulled in at site build time.

## System structure

- [Architecture](architecture.md)
  Top-level module layout and the spec-driven pipeline.
- [Layer Boundaries](layer-boundaries.md)
  What each layer owns and depends on, and the interfaces layers compose through.

## The spec

- [Op Manifest](manifest.md)
  The `src/tileops/manifest/` package as the source of truth for op interfaces.

## Building an op

- [Op Interface Design](ops-design.md)
  Playbook for scaffolding a new op from a manifest entry.
- [Op Interface Reference](ops-design-reference.md)
  Interface tables, codegen, naming, and the family-base protocol.
- [Slot Rules](op-slot-rules.md)
  The authoritative rule, example, and common mistakes per op-file slot.

## Acceptance

- [Testing & Benchmarking](testing.md)
  Separation of correctness tests and profiling benchmarks.
- [Roofline](roofline.md)
  Performance model and the `roofline` manifest field.
