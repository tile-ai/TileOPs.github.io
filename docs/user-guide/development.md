# Development Guide { #development-guide }

## Getting the code { #get-code }

```bash
git clone https://github.com/<your-name>/TileOPs
cd TileOPs
git remote add upstream https://github.com/tile-ai/TileOPs
git fetch upstream
git switch -c <branch> upstream/main
```

`<your-name>` is the GitHub account that holds the fork of
[tile-ai/TileOPs](https://github.com/tile-ai/TileOPs). A development branch starts
from the latest upstream `main`, and the upstream repository is recorded as
`upstream`.

## Setting up an environment { #setup }

### The dev image { #docker }

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace -w /workspace \
  ghcr.io/tile-ai/tileops-runner:<tag>

# inside the container
pip install -e . --no-deps --no-build-isolation
```

- The dev images are published at
  [ghcr.io/tile-ai/tileops-runner](https://github.com/tile-ai/TileOPs/pkgs/container/tileops-runner);
  development uses a tag ending in `-dev`.
- The image carries CUDA, PyTorch, TileLang and the test tools, and comes from the
  same build as the CI's GPU runners.
- `--no-deps` skips resolving and installing dependencies, and uses the ones the
  image already has.

The dev image does not include pre-commit. Run the following on the host, or in
any other environment with Python:

```bash
pip install pre-commit
pre-commit install
```

### A local environment { #local }

```bash
pip install -e '.[dev]' -c constraints.txt
pre-commit install
```

- The supported combination of Python, PyTorch, CUDA, GPU architecture and
  TileLang is the one listed under Prerequisites in the TileOPs
  [README](https://github.com/tile-ai/TileOPs#installation).
- `-c constraints.txt` applies the repository's dependency constraints, so the
  local dependencies match the combination CI validates.

### Checking the environment { #verify }

```bash
python -m pytest -q tests -m smoke
```

Each kernel under test is compiled and cached on its first call, and later
identical calls reuse the cache; see [Kernel compilation and caching](#compile).

## Making a change { #change }

### The relevant spec and design docs { #design-first }

The design docs and the manifest define the constraints on ops, kernels and
tests. When a change to the implementation affects those constraints, the same PR
updates the spec.

| Change | Documents |
| --- | --- |
| Adding an op | [Adding a new op](../new-op.md) |
| An existing op or kernel | [Op Interfaces](../design/ops-design.md), [Slot Rules](../design/op-slot-rules.md) |
| A spec in the manifest | [Spec Fields](manifest/writing.md), [Manifest](../design/manifest.md) |
| Tests | [Testing](../design/testing.md), [Layer Boundaries](../design/layer-boundaries.md) |
| An op's docstring | [Docstrings](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#docstrings) |

### Kernel compilation and caching { #compile }

```python
op = GemmFwdOp()
d = op(a, b)   # first call: compiles the kernel and caches it
d = op(a, b)   # the same call: reuses the cache
```

- Installing TileOPs compiles no kernels. TileLang compiles a kernel the first time
  an op is called.
- An op looks up the entry serving a call by the whole call; what is compiled is
  cached by build identity, and calls that select the same implementation class
  with the same build identity share one entry. TileLang compiles when no entry
  matches. Build identity is defined in
  [How an op selects a kernel](dispatch/index.md).
- Under an editable install, changes to Python source take effect directly; a
  change to dependencies or build configuration needs the install command run
  again.

## Running the tests { #tests }

| Command | Tests included |
| --- | --- |
| `python -m pytest -q tests -m smoke` | `smoke`: the critical path |
| `python -m pytest -q tests -m "smoke or full"` | adds `full`: standard correctness coverage |
| `python -m pytest -q tests -m "smoke or full or nightly"` | adds `nightly`: exhaustive and long-running cases |
| `python -m pytest -q tests` | every test, with no marker filter |

A single test file runs with `python -m pytest -q <test-file>`.

These checks need no GPU:

```bash
python -m pytest -q tests/test_validate_manifest.py   # manifest spec validation
python -m pytest -q benchmarks/tests                  # benchmark infrastructure tests
pre-commit run --all-files                            # lint, the same as CI's pre-commit check
```

## Running benchmarks { #bench }

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev,bench]' -c constraints.txt
python -m pytest -q <bench-file>
```

- A PR that changes a kernel or an op includes benchmark results, compared against
  an implementation outside TileOPs.
- The baseline libraries install through the `bench` extra. The dev image installs
  them at build time too; a baseline that fails to install there does not stop the
  build, and `sgl-kernel` is not in the image. A baseline missing from the
  container is installed at the version the repository declares.
- With `--tileops-verify`, a benchmark only checks correctness and times nothing, which
  is a quick way to confirm the numerics after changing a kernel.
- How to write a benchmark file is in [Writing benchmarks](benchmark/writing.md); how the
  numbers are timed is in [How a benchmark is timed](../timing.md).

## Opening a PR { #pr }

### Title { #pr-title }

| Format | Example |
| --- | --- |
| `[Type] <description>` | `[Doc] Fix the install command in the README` |
| `[Type][Scope] <description>` | `[BugFix][Elementwise] Build the floored tiers at every tuned fold width` |
| `[Type][foundry][Scope] <description>` | `[Perf][foundry][Elementwise] Take the floored tier's reciprocal from one MUFU instruction` |

- CI checks the PR title's format. The values of `Type` are defined in
  [`.claude/conventions/types.sh`](https://github.com/tile-ai/TileOPs/blob/main/.claude/conventions/types.sh).
- `foundry` marks a PR whose kernels were generated by
  [TileFoundry](https://github.com/tile-ai/TileFoundry).

### Description { #pr-body }

The PR description follows the
[PR template](https://github.com/tile-ai/TileOPs/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
When a change touches `tests/`, the description includes the change in the test
count:

```bash
python scripts/test_node_delta.py --base upstream/main
```

### CI { #ci }

- A draft PR runs the PR title check and the manifest statistics only, and skips
  the CPU checks and GPU smoke tests below. They run once the PR is marked
  **Ready for review**.
- The CPU checks are pre-commit, gitleaks, manifest validation, actionlint, the
  compile contract check and the packaging check, plus the benchmark contract tests
  when a change touches the benchmarks.
- The GPU smoke tests run once pre-commit, gitleaks and actionlint pass, scoped by
  the files the PR changes. A PR that changes no Python, manifest or native source
  skips them.

## FAQ { #faq }

### A local install fails while re-resolving CUDA or TileLang { #faq-rebuild }

```bash
PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -c constraints.txt
```

With CUDA and TileLang already installed, turning off build isolation lets the
build use the installed versions.

### Checks that run without an SM90 GPU { #faq-no-gpu }

The three GPU-free checks under [Running the tests](#tests). The tests that need a
GPU run in the PR's CI.
