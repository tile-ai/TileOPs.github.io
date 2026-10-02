# 开发指南 { #development-guide }

## 获取代码 { #get-code }

```bash
git clone https://github.com/<your-name>/TileOPs
cd TileOPs
git remote add upstream https://github.com/tile-ai/TileOPs
git fetch upstream
git switch -c <branch> upstream/main
```

`<your-name>` 是 fork [tile-ai/TileOPs](https://github.com/tile-ai/TileOPs) 所在的 GitHub 账号。开发分支从上游最新的 `main` 创建，上游仓库记录为 `upstream`。

## 搭建开发环境 { #setup }

### 使用 dev 镜像 { #docker }

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace -w /workspace \
  ghcr.io/tile-ai/tileops-runner:<tag>

# 以下命令在容器内执行
pip install -e . --no-deps --no-build-isolation
```

- dev 镜像发布在 [ghcr.io/tile-ai/tileops-runner](https://github.com/tile-ai/TileOPs/pkgs/container/tileops-runner)，开发使用以 `-dev` 结尾的 tag。
- 镜像包含 CUDA、PyTorch、TileLang 与测试工具，与 CI 的 GPU runner 出自同一构建流程。
- `--no-deps` 跳过依赖的解析与安装，直接使用镜像中已有的依赖。

dev 镜像不包含 pre-commit。以下命令在宿主机或另一个已安装 Python 的环境中执行：

```bash
pip install pre-commit
pre-commit install
```

### 使用本地环境 { #local }

```bash
pip install -e '.[dev]' -c constraints.txt
pre-commit install
```

- 支持的 Python、PyTorch、CUDA、GPU 架构与 TileLang 版本组合，以 TileOPs 仓库 [README](https://github.com/tile-ai/TileOPs#installation) 的 Prerequisites 为准。
- `-c constraints.txt` 使用仓库提供的依赖约束，使本地依赖与 CI 验证的组合一致。

### 确认环境可用 { #verify }

```bash
python -m pytest -q tests -m smoke
```

被测的 kernel 在首次调用时编译并写入缓存，之后相同的调用复用缓存，见 [kernel 的编译与缓存](#compile)。

## 修改代码 { #change }

### 相关的 spec 与设计文档 { #design-first }

设计文档与 manifest 定义 op、kernel 与测试的约束。实现的变更影响这些约束时，同一个 PR 同步更新对应的 spec。

| 改动内容 | 相关文档 |
| --- | --- |
| 新增 op | [添加新 op](../new-op.md) |
| 修改已有的 op 或 kernel | [Op 接口](../design/ops-design.md)、[Slot 规则](../design/op-slot-rules.md) |
| 修改 manifest 中的 spec | [写一个 spec](manifest/writing.md)、[manifest 规范](../design/manifest.md) |
| 修改测试 | [测试](../design/testing.md)、[层间边界](../design/layer-boundaries.md) |
| 修改 op 的 docstring | [docstring 的写法](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#docstrings) |

### kernel 的编译与缓存 { #compile }

```python
op = GemmFwdOp()
d = op(a, b)   # 首次调用：编译 kernel 并写入缓存
d = op(a, b)   # 相同的调用：复用缓存
```

- 安装 TileOPs 时不编译 kernel。kernel 在 op 首次被调用时由 TileLang 编译。
- op 按完整的调用信息查找服务这次调用的 entry；编译结果按 build identity 缓存，选中同一实现类且 build identity 相同的调用共用同一个 entry。没有对应的 entry 时，TileLang 执行编译。build identity 的定义见 [op 如何选择 kernel](dispatch/index.md)。
- editable 安装下，Python 源码的改动直接生效；依赖或构建配置变更后，需要重新执行安装命令。

## 运行测试 { #tests }

| 命令 | 包含的测试 |
| --- | --- |
| `python -m pytest -q tests -m smoke` | `smoke`：关键路径 |
| `python -m pytest -q tests -m "smoke or full"` | 另含 `full`：标准的正确性覆盖 |
| `python -m pytest -q tests -m "smoke or full or nightly"` | 另含 `nightly`：穷举与耗时长的用例 |
| `python -m pytest -q tests` | 全部测试，不按 marker 过滤 |

单个测试文件以 `python -m pytest -q <test-file>` 运行。

以下检查不需要 GPU：

```bash
python -m pytest -q tests/test_validate_manifest.py   # manifest spec 校验
python -m pytest -q benchmarks/tests                  # benchmark 基础设施测试
pre-commit run --all-files                            # 代码检查，与 CI 中的 pre-commit 一项相同
```

## 运行 benchmark { #bench }

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev,bench]' -c constraints.txt
python -m pytest -q <bench-file>
```

- 改动 kernel 或 op 的 PR 需附上 benchmark 结果，对比对象是 TileOPs 以外的实现。
- benchmark 的对比库通过 `bench` extra 安装。dev 镜像在构建时也会安装这些库，其中个别库安装失败不会中止构建，`sgl-kernel` 则不在镜像中；容器内缺少的库按仓库声明的版本另行安装。
- 计时方式见 [benchmark 的计时方法](../timing.md)。

## 提交 PR { #pr }

### 标题 { #pr-title }

| 格式 | 例子 |
| --- | --- |
| `[Type] <description>` | `[Doc] Fix the install command in the README` |
| `[Type][Scope] <description>` | `[BugFix][Elementwise] Build the floored tiers at every tuned fold width` |
| `[Type][foundry][Scope] <description>` | `[Perf][foundry][Elementwise] Take the floored tier's reciprocal from one MUFU instruction` |

- CI 检查 PR 标题的格式。`Type` 的取值定义在 [`.claude/conventions/types.sh`](https://github.com/tile-ai/TileOPs/blob/main/.claude/conventions/types.sh) 中。
- `foundry` 标记 kernel 由 [TileFoundry](https://github.com/tile-ai/TileFoundry) 生成的 PR。

### 描述 { #pr-body }

PR 描述按 [PR 模板](https://github.com/tile-ai/TileOPs/blob/main/.github/PULL_REQUEST_TEMPLATE.md)填写。改动涉及 `tests/` 时，描述中附上测试用例数量的变化：

```bash
python scripts/test_node_delta.py --base upstream/main
```

### CI { #ci }

- Draft PR 只运行 PR 标题检查与 manifest 统计，跳过下面的 CPU 检查与 GPU smoke 测试。PR 转为 **Ready for review** 后，这些检查才会运行。
- CPU 检查包括 pre-commit、gitleaks、manifest 校验、actionlint、编译约定检查与打包检查；改动涉及 benchmark 时，另含 benchmark 约定测试。
- GPU smoke 测试在 pre-commit、gitleaks 与 actionlint 通过后运行，测试范围由改动的文件决定。没有改动 Python、manifest 或原生代码的 PR 跳过 GPU smoke。

## 常见问题 { #faq }

### 本地安装在重新解析 CUDA 或 TileLang 时失败 { #faq-rebuild }

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -c constraints.txt
```

本机已安装 CUDA 与 TileLang 时，关闭构建隔离，构建过程直接使用已安装的版本。

### 没有 SM90 GPU 时可运行的检查 { #faq-no-gpu }

[运行测试](#tests)一节中不需要 GPU 的三项检查。需要 GPU 的测试由 PR 上的 CI 运行。
