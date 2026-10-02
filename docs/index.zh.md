# TileOPs

TileOPs 是一个面向大模型推理的探索性算子库，构建在 [TileLang](https://github.com/tile-ai/tilelang) 之上。TileOPs 为 agent 设计，整个项目由 agent 构建。这样的项目需要满足明确的代码质量要求：项目结构保持一致，不随 op 增多而发散或膨胀，代码可以长期维护。TileOPs 的设计围绕三个目标：

- **可维护**：每个 op 由一份 spec 声明，agent 依据 spec 生成实现。同一 family 的 op 共用一套接口与规则，新增的 op 和 kernel 遵循这套接口与规则。
- **可验证**：spec 指定正确性所依据的参考实现。测试将 kernel 的输出与参考实现比较，性能评测将实测性能与 roofline 模型给出的上界比较。验收标准事先确定，检查过程自动执行。
- **可调优**：roofline 模型给出每个 kernel 与性能上界之间的差距。[每晚的 benchmark](benchmarks/index.md) 在同一硬件上将每个 kernel 与最快的其他实现比较。

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

- [博客](blog/index.md)：TileOPs 开发中的技术探索。
- [用户指南](user-guide/index.md)：读写 manifest、接入 `torch.compile`、benchmark 的计时方法、接入新硬件 backend。
- [API 参考](api/index.md)：各 op family 的构造参数与调用方式。
- [性能数据](benchmarks/index.md)：每晚在 H200 上实测，逐个 workload 与其他实现对比。

## 相关链接

- [GitHub](https://github.com/tile-ai/TileOPs)
- [开发指南](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
