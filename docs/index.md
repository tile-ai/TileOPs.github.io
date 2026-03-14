# TileOPs

High-performance LLM operator library built on [TileLang](https://github.com/tile-ai/tilelang).

## Installation

```bash
pip install tileops
```

## Quick Start

```python
import torch
from tileops.ops import GemmOp

op = GemmOp(dtype=torch.float16)
C = op.forward(A, B)
```

## Links

- [GitHub](https://github.com/tile-ai/TileOPs)
- [Contributing](https://github.com/tile-ai/TileOPs/blob/main/docs/CONTRIBUTING.md)
- [Development Guide](https://github.com/tile-ai/TileOPs/blob/main/docs/DEVELOPMENT.md)
