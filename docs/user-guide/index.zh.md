# 用户指南

## 开始开发

- [开发指南](development.md)
  获取代码，搭建 dev 镜像或本地环境，运行测试，提交 PR。

## spec 与 op

- [读写 manifest](manifest/index.md)
  系统如何使用 spec，描述 spec 所用的概念，以及 spec 的写法。
- [添加新 op](../new-op.md)
  从一份 spec 到 `status: implemented` 的六个步骤。

## kernel 与硬件

- [op 如何选择 kernel](dispatch/index.md)
  op 选中 kernel 的过程，新增 kernel 的方法，以及 backend 的接入方式。
- [接入新硬件 backend](../backends.md)
  在某一类设备上，用自己的 kernel 实现 op。

## 集成与性能

- [接入 torch.compile](../torch-compile.md)
  op 在编译图中的形态，以及调用时的约定。
- [benchmark 的计时方法](../timing.md)
  性能数据页上的数字如何测得。
- [编写 benchmark](benchmark/writing.md)
  用 manifest case 校验并计时 op 与对比实现。
