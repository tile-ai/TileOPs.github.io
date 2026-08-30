#!/usr/bin/env bash
# Fetch the latest nightly benchmark snapshot from tile-ai/TileOPs-nightly and
# regenerate docs/benchmarks/index.md.
#
# index.md is a build artifact, not source: the committed file is a placeholder.
# Both deploy.yml and render-benchmarks.yml call this before `mkdocs gh-deploy`.
#
# `gh-deploy --force` republishes the whole site, so a bad render must not
# overwrite the live page: nothing published yet keeps the placeholder and
# succeeds, anything else aborts the deploy.
#
# Needs python3 and pyyaml. A ./TileOPs checkout supplies the spec manifest the
# shapes are read from; without it each workload is named by its benchmark id.
set -euo pipefail

# One commit per run on `snapshots`; the newest is what this renders, and
# `git log snapshots` is where an older one is read back from.
snapshots="https://github.com/tile-ai/TileOPs-nightly"
base="https://raw.githubusercontent.com/tile-ai/TileOPs-nightly/snapshots"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fetch() {  # fetch <name> <dest>; prints the HTTP status, 000 if it never got one
  curl -sS -L --retry 3 --retry-delay 2 --retry-all-errors \
    -o "$2" -w '%{http_code}' "$base/$1" 2>/dev/null || echo 000
}

# 404 means nothing has been published yet, and is the one status that may keep
# the placeholder and succeed.
code="$(fetch bench_results.xml "$work/bench_results.xml")"
if [ "$code" = "404" ]; then
  echo "::warning::${snapshots} has published no snapshot yet; keeping placeholder benchmark page"
  exit 0
elif [ "$code" != "200" ]; then
  echo "::error::fetching bench_results.xml answered ${code}; aborting so the live benchmark page is not overwritten"
  exit 1
fi
if [ "$(fetch meta.json "$work/meta.json")" != "200" ]; then
  echo "::error::the snapshot has no meta.json; aborting rather than publishing numbers with no environment"
  exit 1
fi
if [ "$(fetch test_results.xml "$work/test_results.xml")" != "200" ]; then
  echo "::warning::test_results.xml not published; rendering without test status"
  rm -f "$work/test_results.xml"
fi

read_meta() {  # read_meta <key>; missing key -> "unknown"
  python3 -c "import json;print(json.load(open('$work/meta.json')).get('$1','unknown'))"
}
bench_commit="$(read_meta commit)"
bench_date="$(read_meta date)"
bench_gpu="$(read_meta gpu)"
rendered="$(date -u +'%Y-%m-%d %H:%M UTC')"

test_arg=()
[ -f "$work/test_results.xml" ] && test_arg=(--test-xml "$work/test_results.xml")

# The workload shapes on the pages come from the spec manifest, so they must be
# the manifest as it stood when the benchmark ran: the ./TileOPs checkout is at
# main, which is ahead of the snapshot's commit and may have moved a shape under
# a label since. Read the manifest out of that commit's tree instead. Failing
# that, gen_bench_pages.py falls back to the checkout, which is right for every
# label the two commits agree on and wrong only where the shapes moved.
manifest_arg=()
manifest_dir="$work/manifest"
if [ -d TileOPs/.git ] && [ "$bench_commit" != "unknown" ]; then
  if git -C TileOPs fetch --quiet --depth 1 origin "$bench_commit" 2>/dev/null; then
    mkdir -p "$manifest_dir"
    n=0
    while read -r path; do
      [ -n "$path" ] || continue
      git -C TileOPs show "$bench_commit:$path" > "$manifest_dir/$(basename "$path")" && n=$((n + 1))
    done < <(git -C TileOPs ls-tree --name-only "$bench_commit" src/tileops/manifest/ \
             | grep '\.yaml$' || true)
    if [ "$n" -gt 0 ]; then
      manifest_arg=(--manifest-dir "$manifest_dir")
      echo "read $n manifest files from TileOPs ${bench_commit:0:12}"
    fi
  fi
fi
if [ ${#manifest_arg[@]} -eq 0 ]; then
  echo "::warning::could not read the manifest at ${bench_commit:0:12}; workload shapes come from the ./TileOPs checkout instead"
fi

python3 scripts/gen_bench_pages.py \
  --bench-xml "$work/bench_results.xml" \
  ${test_arg[@]+"${test_arg[@]}"} \
  ${manifest_arg[@]+"${manifest_arg[@]}"} \
  --meta "$work/meta.json" \
  --commit "$bench_commit" --date "$bench_date" --gpu "$bench_gpu" --rendered "$rendered"
