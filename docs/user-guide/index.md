# User Guide

| Page | Contents |
| --- | --- |
| [Reading and writing the manifest](manifest/index.md) | how the system uses a spec, the concepts used to describe a spec, and how to write one |
| [How an op selects a kernel](dispatch/index.md) | how an op selects a kernel, how to add a kernel, and how a backend joins |
| [Adding a new op](../new-op.md) | the six steps from a spec to `status: implemented` |
| [Bringing an op into torch.compile](../torch-compile.md) | what an op looks like inside a compiled graph, and the conventions a caller follows |
| [How a benchmark is timed](../timing.md) | how the numbers on the Benchmarks pages are measured |
| [Adding a hardware backend](../backends.md) | serving the ops on one class of devices with your own kernels |
