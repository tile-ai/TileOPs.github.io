# TileOPs

TileOPs 是一个面向大模型推理的探索性算子库，构建在 [TileLang](https://github.com/tile-ai/tilelang) 之上。TileOPs 为 agent 设计，整个项目由 agent 构建。这样的项目需要满足明确的代码质量要求：项目结构保持一致，不随 op 增多而发散或膨胀，代码可以长期维护。TileOPs 的设计围绕三个目标：

- **可维护**：每个 op 由一份 spec 声明，agent 依据 spec 生成实现。同一 family 的 op 共用一套接口与规则，新增的 op 和 kernel 遵循这套接口与规则。
- **可验证**：spec 指定正确性所依据的参考实现。测试将 kernel 的输出与参考实现比较，性能评测将实测性能与 roofline 模型给出的上界比较。验收标准事先确定，检查过程自动执行。
- **可调优**：roofline 模型给出每个 kernel 与性能上界之间的差距。[每晚的 benchmark](benchmarks/index.md) 在同一硬件上将每个 kernel 与最快的其他实现比较。

## 安装

```bash
pip install tileops
```

每次发布在以下一组配置上验证：

- Linux
- Python 3.12 或 3.13
- PyTorch 2.13
- [TileLang](https://github.com/tile-ai/tilelang) 0.1.12
- CUDA Toolkit 13.2，且 `nvcc` 在 `PATH` 上：TileLang 在运行时编译 kernel
- 计算能力 9.0（SM90）的 GPU，在 H200 上测试

部分 kernel 另有 SM80、SM86 或 SM89 路径，不在发布验证范围内。从源码构建见[开发指南](user-guide/development.md)。

## 快速开始

op 在构造时不绑定任何形状。形状和 dtype 取自调用时传入的张量，特化后的 kernel 在首次调用时编译并缓存。

```python
import torch
from tileops.gemm import GEMMFwdOp

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

op = GEMMFwdOp()                        # 默认 NT 布局：a=[M, K], b=[N, K]
d = op(a, b)                         # -> [M, N]
flops, nbytes = op.eval_roofline()   # 本次调用所需的计算量与访存量
```

## 后续阅读

- [博客](blog/index.md)：TileOPs 开发中的技术探索。
- [用户指南](user-guide/index.md)：读写 manifest、接入 `torch.compile`、benchmark 的计时方法、接入新硬件 backend。
- [API 参考](api/index.md)：各 op family 的构造参数与调用方式。
- [API 稳定性](stability.md)：哪些名字是公开 API，以及一次发布可以改动什么。
- [性能数据](benchmarks/index.md)：每晚在 H200 上实测，逐个 workload 与其他实现对比。

## 相关链接

- [GitHub](https://github.com/tile-ai/TileOPs)
- [PyPI](https://pypi.org/project/tileops/)
