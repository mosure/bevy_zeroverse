#!/usr/bin/env bash
set -euo pipefail

# Read the index so the same check works before committing and on a CI checkout.
generated=$(git ls-files --cached --ignored --exclude-standard -- docs tex www)
if [[ -n "$generated" ]]; then
    printf 'Generated publication outputs belong in the Pages artifact:\n%s\n' "$generated" >&2
    exit 1
fi

size=$(git ls-tree -rl "$(git write-tree)" | awk '$2 == "blob" { total += $4 } END { printf "%.0f", total }')
budget=$((64 * 1024 * 1024))
printf 'Tracked source checkout: %s bytes (budget %s bytes)\n' "$size" "$budget"
if (( size > budget )); then
    echo 'Inspect new assets or study outputs with git ls-tree -rl. See docs/project_page.md.' >&2
    exit 1
fi
