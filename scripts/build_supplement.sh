#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

cd "$EKT_ROOT"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "ERROR: commit or remove working-tree changes before building the supplement." >&2
  exit 1
fi

archive_name="KnowinRouter-Source"
output_dir="${1:-$EKT_ROOT/dist}"
output_path="$output_dir/$archive_name.zip"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

mkdir -p "$build_dir/$archive_name" "$output_dir"
git archive HEAD | tar -x -C "$build_dir/$archive_name"

(
  cd "$build_dir/$archive_name"
  find . -type f ! -name MANIFEST.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > MANIFEST.sha256
)

rm -f "$output_path"
(
  cd "$build_dir"
  zip -q -9 -r "$output_path" "$archive_name"
)

archive_size="$(stat -c %s "$output_path")"
max_size=$((50 * 1024 * 1024))
if (( archive_size > max_size )); then
  rm -f "$output_path"
  echo "ERROR: supplement exceeds the 50 MiB limit." >&2
  exit 1
fi

(
  cd "$output_dir"
  sha256sum "$archive_name.zip" > "$archive_name.zip.sha256"
)
printf 'Created %s (%s bytes)\n' "$output_path" "$archive_size"
