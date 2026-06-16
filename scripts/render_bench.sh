#!/usr/bin/env bash
# Fetch the latest nightly benchmark snapshot from the public TileOPs
# `nightly-bench` orphan branch and regenerate docs/benchmarks/index.md.
#
# index.md is a build artifact, not source: the committed file is a placeholder.
# Both deploy.yml (push) and render-benchmarks.yml (schedule) call this before
# `mkdocs gh-deploy`, so every deploy serves fresh data. If the snapshot is
# unavailable, the placeholder is left in place and the build still succeeds.
#
# Requires TileOPs checked out at ./TileOPs (for scripts/nightly_report.py,
# loaded by gen_bench_pages.py) and python on PATH.
set -euo pipefail

base="https://raw.githubusercontent.com/tile-ai/TileOPs/nightly-bench"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if ! curl -fsSL "$base/bench_results.xml" -o "$work/bench_results.xml"; then
  echo "::warning::nightly-bench snapshot unavailable; keeping placeholder benchmark page"
  exit 0
fi
curl -fsSL "$base/meta.json" -o "$work/meta.json"
curl -fsSL "$base/test_results.xml" -o "$work/test_results.xml" \
  || echo "::warning::test_results.xml not found on nightly-bench; rendering without test status"

commit=$(python -c "import json;print(json.load(open('$work/meta.json'))['commit'])")
date=$(python -c "import json;print(json.load(open('$work/meta.json'))['date'])")
gpu=$(python -c "import json;print(json.load(open('$work/meta.json'))['gpu'])")
rendered=$(date -u +'%Y-%m-%d %H:%M UTC')

test_arg=()
[ -f "$work/test_results.xml" ] && test_arg=(--test-xml "$work/test_results.xml")

python scripts/gen_bench_pages.py \
  --bench-xml "$work/bench_results.xml" \
  ${test_arg[@]+"${test_arg[@]}"} \
  --commit "$commit" --date "$date" --gpu "$gpu" --rendered "$rendered"
