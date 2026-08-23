# TileOPs

TileOPs 是一个面向大模型推理的算子库，构建在 [TileLang](https://github.com/tile-ai/tilelang) 之上，同一套算子接口可以由不同后端在不同硬件上实现。

它与手写算子库的不同之处在于组织方式：每个算子先以一份 spec 声明，再由 agent 依据这份 spec 生成实现。spec 既是代码生成的唯一依据，也是验收的标准 —— 正确性对照 spec 指定的参考实现，性能对照 roofline 模型给出的上界，两项都不依赖人的判断。因此一个实现可以随时从 spec 重新生成，而反过来做不到。

对使用者而言，它就是一批可以直接调用的算子：形状与 dtype 在调用时确定，特化后的 kernel 在首次使用时自动调优并缓存，随后可以与 CUDA graph 配合使用；每个算子各自声明是否支持 `torch.compile(fullgraph=True)`。

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

## 从这里继续

- [接入 torch.compile](torch-compile.md) —— 算子在编译图中的形态，以及调用时需要遵守的约定
- [接入新硬件后端](backends.md) —— 用自己的 kernel 接管某一类设备上的算子
- [API 参考](api/linear-algebra.md) —— 各算子族的构造参数与调用方式
- [性能数据](benchmarks/index.md) —— 每晚在 H200 上与其他实现同台测量的结果

## 相关链接

- [GitHub](https://github.com/tile-ai/TileOPs)
- [开发指南](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
