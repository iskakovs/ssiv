#!/usr/bin/env bash
# Download the Autor-Dorn-Hanson data distributed with the ShiftShareSE R package
# and convert it to CSV for the validation tests and the replication example.
# Requires curl and R (base R only).
# Usage: bash scripts/fetch_adh.sh <output dir>
set -euo pipefail
out="${1:?usage: fetch_adh.sh <output dir>}"
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
curl -fsSL -o "$tmp/ADH.rda" \
  https://raw.githubusercontent.com/kolesarm/ShiftShareSE/master/data/ADH.rda
Rscript --vanilla "$here/export_adh.R" "$tmp/ADH.rda" "$out"
rm -rf "$tmp"
