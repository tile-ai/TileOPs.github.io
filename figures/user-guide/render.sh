#!/usr/bin/env bash
# Render the user-guide figures into docs/user-guide/<guide>/img/.
#
#   figures/user-guide/render.sh
#
# <name>.zh.puml becomes img/<name>.zh.svg, the Chinese page's figure, and <name>.en.puml
# becomes img/<name>.svg, the English one; manifest/overview.py draws both languages. The
# PlantUML release is pinned because releases lay diagrams out differently.
set -euo pipefail
cd "$(dirname "$0")"

VERSION="1.2026.6"
JAR="${PLANTUML_JAR:-$HOME/.local/lib/plantuml.jar}"
if ! java -jar "$JAR" -version 2>/dev/null | grep -q "PlantUML version ${VERSION}"; then
  echo "PlantUML ${VERSION} not found at ${JAR}; set PLANTUML_JAR. It also needs graphviz and Noto Sans SC." >&2
  exit 1
fi

DOCS=../../docs/user-guide
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for src in */*.puml; do
  guide="$(dirname "$src")"
  base="$(basename "$src" .puml)"   # <name>.zh or <name>.en
  name="${base%.*}"
  lang="${base##*.}"
  suffix=""
  if [ "$lang" = zh ]; then suffix=".zh"; fi
  out="$DOCS/$guide/img/$name$suffix.svg"
  rm -rf "$tmp/out" && mkdir -p "$tmp/out"
  java -jar "$JAR" -tsvg -failfast2 -o "$tmp/out" "$src"
  # PlantUML writes one font-family and no trailing newline.
  sed "s/font-family=\"'Noto Sans SC'\"/font-family=\"'Noto Sans SC','PingFang SC','Microsoft YaHei',sans-serif\"/g" \
    "$tmp"/out/*.svg >"$out"
  if [[ -n "$(tail -c1 "$out")" ]]; then printf '\n' >>"$out"; fi
  echo "$out"
done

python3 manifest/overview.py zh "$DOCS/manifest/img/overview.zh.svg"
python3 manifest/overview.py en "$DOCS/manifest/img/overview.svg"
echo "$DOCS/manifest/img/overview.{zh.,}svg"
