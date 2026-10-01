# TileOPs

TileOPs 是一个面向大模型推理的算子库，构建在 [TileLang](https://github.com/tile-ai/tilelang) 之上。同一套 op 接口可以由不同 backend 在不同硬件上实现。

TileOPs 与手写算子库的区别在于组织方式：每个 op 先以一份 spec 声明，再由 agent 依据这份 spec 生成实现。spec 是代码生成的唯一依据，也是验收的标准：

- 正确性以 spec 指定的参考实现为准；
- 性能以 roofline 模型给出的上界为准。

两项验收都不依赖人的判断。因此一个实现可以随时从 spec 重新生成，spec 却不能从实现反推。

对使用者而言，TileOPs 提供一批可以直接调用的 op。形状与 dtype 在调用时确定；特化后的 kernel 在首次使用时构造并缓存，之后可以与 CUDA graph 配合使用。每个 op 各自声明是否支持 `torch.compile(fullgraph=True)`。

## 安装

```bash
pip install tileops
```

## 快速开始

op 在构造时不绑定任何形状。形状和 dtype 取自调用时传入的张量，特化后的 kernel 在首次调用时编译并缓存。

```python
import torch
from tileops.gemm import GemmFwdOp

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

op = GemmFwdOp()                        # 默认 NT 布局：a=[M, K], b=[N, K]
d = op(a, b)                         # -> [M, N]
flops, nbytes = op.eval_roofline()   # 本次调用所需的计算量与访存量
```

## 后续阅读

- [使用指南](user-guide/index.md)：读写 manifest、接入 `torch.compile`、benchmark 的计时方法、接入新硬件 backend。
- [API 参考](api/index.md)：各 op family 的构造参数与调用方式。
- [性能数据](benchmarks/index.md)：每晚在 H200 上实测，逐个 workload 与其他实现对比。

## 相关链接

- [GitHub](https://github.com/tile-ai/TileOPs)
- [开发指南](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
