# TileOPs

跨后端的 LLM 算子库，由 spec 驱动、由 agent 编写。一切以 spec 为准：kernel 从 spec 推导而来，写得对不对也由 spec 判定。

## 安装

```bash
pip install tileops
```

## 快速开始

算子在构造时什么都不固定。形状和 dtype 由调用时传入的张量决定，特化过的 kernel 在首次调用时编译，之后的调用直接取缓存。

```python
import torch
from tileops.ops import GemmOp

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

op = GemmOp()                        # 默认 NT 布局：a=[M, K], b=[N, K]
d = op(a, b)                         # -> [M, N]
flops, nbytes = op.eval_roofline()   # 这次调用算了多少、搬了多少
```

## 相关链接

- [性能数据](benchmarks/index.md) —— 每个算子每晚在 H200 上与其他实现同台测量
- [GitHub](https://github.com/tile-ai/TileOPs)
- [开发指南](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
