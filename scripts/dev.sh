#!/usr/bin/env bash
# Build and preview the site locally. A first run needs a virtualenv and the
# ./TileOPs checkout; both are set up below, gitignored, and reused after that.
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
    --port)
      case "${2:-}" in
        "" | *[!0-9]*) echo "--port needs a number" >&2; exit 2 ;;
      esac
      port="$2"; shift 2 ;;
    --no-venv) use_venv=0; shift ;;
    --update) update=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# The checkout mkdocstrings imports `tileops` from, sparse over the paths the
# workflows check out. An existing ./TileOPs — a symlink to a clone elsewhere,
# say — is left alone. It stays at the revision it was cloned at while CI reads
# current upstream, so an op renamed there fails CI a local build had passed.
if [ -L TileOPs ] && [ ! -e TileOPs ]; then
  # `-e` follows the link, so a symlink whose target is gone reads as missing,
  # and the clone below would fail on a path that already exists.
  echo "./TileOPs is a symlink to $(readlink TileOPs), which is not there;" \
       "repoint it or remove it" >&2
  exit 1
elif [ ! -e TileOPs ]; then
  echo "==> cloning tile-ai/TileOPs into ./TileOPs"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/tile-ai/TileOPs.git TileOPs
  git -C TileOPs sparse-checkout set --no-cone \
    docs/design docs/perf src/tileops scripts/nightly_report.py
elif [ "$update" = 1 ]; then
  echo "==> updating ./TileOPs"
  git -C TileOPs pull --ff-only
fi

py="$(command -v python3 || command -v python)"  # a system may ship only one

if [ "$use_venv" = 1 ]; then
  venv=".venv"
  if [ ! -x "$venv/bin/python" ]; then
    echo "==> creating $venv"
    "$py" -m venv "$venv"
  fi
  export PATH="$PWD/$venv/bin:$PATH"
  py="$venv/bin/python"

  # Reinstall only when requirements-docs.txt is newer than the last install, so
  # a serve starts at once. Nothing there is pinned, so the versions stay at
  # whatever that install resolved: delete .venv to move them.
  stamp="$venv/.docs-requirements"
  if [ ! -f "$stamp" ] || [ requirements-docs.txt -nt "$stamp" ]; then
    echo "==> installing requirements-docs.txt"
    "$py" -m pip install --quiet --upgrade pip
    "$py" -m pip install --quiet -r requirements-docs.txt
    touch "$stamp"
  fi
fi

if ! command -v mkdocs > /dev/null; then
  echo "mkdocs is not on PATH; drop --no-venv, or pip install -r requirements-docs.txt" >&2
  exit 1
fi
# Which ops the api/ pages name is written by hand, so an op renamed upstream
# shows up here as a report rather than as `Could not collect` mid-build.
"$py" scripts/check_api_pages.py

case "$cmd" in
  bench)
    bash scripts/render_bench.sh
    # Every page under docs/benchmarks/ is gitignored except index.md, whose
    # committed copy is the placeholder the render has just overwritten.
    echo "==> restore the placeholder before committing:"
    echo "    git checkout docs/benchmarks/index.md"
    ;;
  build)
    # checks.yml's rule: every warning fails except griffe's, which are TileOPs'
    # docstrings rather than this repository's.
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
