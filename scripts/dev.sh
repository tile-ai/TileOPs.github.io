#!/usr/bin/env bash
# Build and preview the site locally, with the same dependencies the deploy uses.
#
# The two things a first run needs are set up here rather than by hand: a
# virtualenv holding requirements-docs.txt, and the ./TileOPs checkout that
# api/ and design/ read. Both are gitignored and reused by later runs.
#
#   bash scripts/dev.sh serve            # install what is missing, then serve
#   bash scripts/dev.sh build            # one-shot build into site/, warnings fail
#   bash scripts/dev.sh bench            # render docs/benchmarks/ from the nightly
#   bash scripts/dev.sh serve --update   # pull ./TileOPs first
#
# --no-venv runs against the interpreter already on PATH, for a shell that
# manages its own environment (conda, an activated venv, uv).
set -euo pipefail

cd "$(dirname "$0")/.."

cmd="serve"
case "${1:-}" in
  serve | build | bench) cmd="$1"; shift ;;
  -*) ;;
  "") ;;
  *) echo "unknown command: $1 (expected serve, build or bench)" >&2; exit 2 ;;
esac

port=8000
use_venv=1
update=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    --no-venv) use_venv=0; shift ;;
    --update) update=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# The checkout mkdocstrings imports `tileops` from. Sparse, like the workflows'
# checkout: the whole repository is not needed to read docstrings and design
# docs. A symlink to a clone elsewhere is left alone, ./TileOPs being whatever
# the developer pointed it at.
if [ ! -e TileOPs ]; then
  echo "==> cloning tile-ai/TileOPs into ./TileOPs"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/tile-ai/TileOPs.git TileOPs
  git -C TileOPs sparse-checkout set --no-cone \
    docs/design docs/perf src/tileops scripts/nightly_report.py
elif [ "$update" = 1 ]; then
  echo "==> updating ./TileOPs"
  git -C TileOPs pull --ff-only
fi

if [ "$use_venv" = 1 ]; then
  venv=".venv"
  if [ ! -x "$venv/bin/python" ]; then
    echo "==> creating $venv"
    python3 -m venv "$venv"
  fi
  export PATH="$PWD/$venv/bin:$PATH"

  # Reinstall only when the requirements file changed, so a serve after an edit
  # starts at once. The stamp holds the checksum the last install ran on.
  stamp="$venv/.docs-requirements"
  want="$(shasum requirements-docs.txt | cut -d' ' -f1)"
  if [ "$(cat "$stamp" 2>/dev/null || true)" != "$want" ]; then
    echo "==> installing requirements-docs.txt"
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements-docs.txt
    echo "$want" > "$stamp"
  fi
fi

if ! command -v mkdocs > /dev/null; then
  echo "mkdocs is not on PATH; drop --no-venv, or pip install -r requirements-docs.txt" >&2
  exit 1
fi

# Which ops the api/ pages name is written by hand, so an op renamed upstream
# shows up here as a report rather than as `Could not collect` mid-build.
python scripts/check_api_pages.py

case "$cmd" in
  bench)
    bash scripts/render_bench.sh
    ;;
  build)
    # The deploy's rule: every warning fails except griffe's, which are TileOPs'
    # docstrings rather than this repository's.
    set -o pipefail
    mkdocs build 2>&1 | tee build.log
    if grep -E '^WARNING' build.log | grep -v 'griffe:'; then
      echo "mkdocs reported the warnings above" >&2
      exit 1
    fi
    ;;
  serve)
    # site_url carries a subpath, so / is not the page — print the one that is.
    echo "==> http://127.0.0.1:${port}/TileOPs.github.io/"
    mkdocs serve --dev-addr "127.0.0.1:${port}"
    ;;
esac
