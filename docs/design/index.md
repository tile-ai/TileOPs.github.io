# Design

Architecture and design documentation for TileOPs internals. The pages
below mirror `docs/design/` in the [`tile-ai/TileOPs`](https://github.com/tile-ai/TileOPs)
repository — the source of truth — pulled in at site build time.

- [Architecture](architecture.md) — top-level module layout and the spec-driven pipeline.
- [Op Manifest](manifest.md) — the `tileops/manifest/` package as the source of truth for op interfaces.
- [Op Interface Design](ops-design.md) — playbook for scaffolding a new op from a manifest entry.
- [Op Interface Reference](ops-design-reference.md) — slot-keyed authoritative rules.
- [Roofline](roofline.md) — performance model and the `roofline` manifest field.
- [Testing & Benchmarking](testing.md) — separation of correctness tests and profiling benchmarks.
- [Trust Model](trust-model.md) — stage boundaries and the guarantees each stage owns.
- [Elementwise Operators](elementwise.md) — kernel strategies, broadcast coalescing, fp8 support.
