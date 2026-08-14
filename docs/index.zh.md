# TileOPs

跨后端的 LLM 算子库，由 spec 驱动、由 agent 编写。spec 是唯一依据：kernel 由它推导，也由它裁定。

## 安装

```bash
pip install tileops
```

## 快速开始

算子在构造时不绑定任何形状。形状和 dtype 取自调用传入的张量，特化后的 kernel 于首次调用时编译并缓存。

```python
import torch
from tileops.ops import GemmOp

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

op = GemmOp()                        # 默认 NT 布局：a=[M, K], b=[N, K]
d = op(a, b)                         # -> [M, N]
flops, nbytes = op.eval_roofline()   # 本次调用所需的计算量与访存量
```

## 相关链接

- [性能数据](benchmarks/index.md) —— 每晚在 H200 上将每个算子与其他实现同台测量
- [GitHub](https://github.com/tile-ai/TileOPs)
- [开发指南](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
