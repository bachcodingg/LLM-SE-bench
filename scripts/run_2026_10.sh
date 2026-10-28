#!/usr/bin/env bash
# The October 2026 re-run: real Docker execution, two tiers from two vendors.
#
#   mid tier:   claude-sonnet-5-5, gpt-5.6-terra
#   cheap tier: claude-haiku-5-5,  gpt-5.6-luna
#
# Gemini is absent because the key used had only free-tier quota (20
# requests a day for gemini-3.6-flash). Add `-m gemini-3.6-flash` on a
# billed key; the client and its pricing are in place.
#
# Usage: scripts/run_2026_10.sh [model ...]   (default: all four)
set -u
cd "$(dirname "$0")/.."
models=("$@")
if [ ${#models[@]} -eq 0 ]; then
  models=(claude-sonnet-5-5 gpt-5.6-terra claude-haiku-5-5 gpt-5.6-luna)
fi
flags=()
for m in "${models[@]}"; do flags+=(-m "$m"); done
for ds in humaneval mbpp defects4j godclass; do
  python cli.py run --dataset "$ds" "${flags[@]}" --runs 3 --max-concurrency 3 --output-dir results
done
