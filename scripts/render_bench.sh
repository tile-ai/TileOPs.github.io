#!/usr/bin/env bash
# Fetch the latest nightly benchmark snapshot from the public TileOPs
# `nightly-bench` orphan branch and regenerate docs/benchmarks/index.md.
#
# index.md is a build artifact, not source: the committed file is a placeholder.
# Both deploy.yml (push) and render-benchmarks.yml (schedule) call this before
# `mkdocs gh-deploy`, so every deploy serves fresh data.
#
# Failure policy (gh-deploy --force republishes the whole site, so a bad render
# must not overwrite the live page):
#   * nightly-bench branch absent  -> no snapshot has ever existed; keep the
#     placeholder and succeed (first-deploy bootstrap).
#   * branch present but fetch fails -> transient 404/network error; exit
#     non-zero so the deploy aborts and the live page is left intact.
#
# Requires TileOPs checked out at ./TileOPs (for scripts/nightly_report.py,
# loaded by gen_bench_pages.py) and python on PATH.
set -euo pipefail

repo="https://github.com/tile-ai/TileOPs"
base="https://raw.githubusercontent.com/tile-ai/TileOPs/nightly-bench"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# `git ls-remote --exit-code` returns 0 when the ref is found, 2 when there is
# no matching ref, and other codes (e.g. 128) for transport/repository errors.
# Only the explicit no-matching-ref case may bootstrap with the placeholder;
# treat transport errors as transient and abort so the live page is preserved.
set +e
git ls-remote --exit-code --heads "$repo" nightly-bench >/dev/null 2>&1
ls_status=$?
set -e
if [ "$ls_status" -eq 2 ]; then
  echo "::warning::nightly-bench branch does not exist yet; keeping placeholder benchmark page"
  exit 0
elif [ "$ls_status" -ne 0 ]; then
  echo "::error::could not query nightly-bench (git ls-remote exit ${ls_status}); aborting so the live benchmark page is not overwritten"
  exit 1
fi

fetch() {  # fetch <remote-name> <dest>; retries to ride out transient errors
  curl -fsSL --retry 3 --retry-delay 2 --retry-all-errors "$base/$1" -o "$2"
}

if ! fetch bench_results.xml "$work/bench_results.xml" || ! fetch meta.json "$work/meta.json"; then
  echo "::error::nightly-bench exists but its snapshot could not be fetched; aborting so the live benchmark page is not overwritten with a placeholder"
  exit 1
fi
fetch test_results.xml "$work/test_results.xml" \
  || echo "::warning::test_results.xml not found on nightly-bench; rendering without test status"

read_meta() {  # read_meta <key>; missing key -> "unknown"
  python -c "import json;print(json.load(open('$work/meta.json')).get('$1','unknown'))"
}
bench_commit="$(read_meta commit)"
bench_date="$(read_meta date)"
bench_gpu="$(read_meta gpu)"
rendered="$(date -u +'%Y-%m-%d %H:%M UTC')"

test_arg=()
[ -f "$work/test_results.xml" ] && test_arg=(--test-xml "$work/test_results.xml")

python scripts/gen_bench_pages.py \
  --bench-xml "$work/bench_results.xml" \
  ${test_arg[@]+"${test_arg[@]}"} \
  --commit "$bench_commit" --date "$bench_date" --gpu "$bench_gpu" --rendered "$rendered"
