# User Guide

## Getting started

- [Development guide](development.md)
  Getting the code, setting up the dev image or a local environment, running the
  tests, and opening a PR.

## Specs and ops

- [Reading and writing the manifest](manifest/index.md)
  How the system uses a spec, the concepts a spec is described in, and how to
  write one.
- [Adding a new op](../new-op.md)
  The six steps from a spec to `status: implemented`.

## Kernels and hardware

- [How an op selects a kernel](dispatch/index.md)
  How an op selects a kernel, how to add a kernel, and how a backend joins.
- [Adding a hardware backend](../backends.md)
  Serving the ops on one class of devices with your own kernels.

## Integration and performance

- [Bringing an op into torch.compile](../torch-compile.md)
  What an op looks like inside a compiled graph, and the conventions a caller
  follows.
- [How a benchmark is timed](../timing.md)
  How the numbers on the Benchmarks pages are measured.
- [Writing benchmarks](benchmark/writing.md)
  Checking and timing an op against other implementations over its manifest cases.
