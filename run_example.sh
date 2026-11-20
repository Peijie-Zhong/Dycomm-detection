#!/usr/bin/env bash
# Run dycommu-tgat.py on the bundled synthetic link stream demo.

set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

"$PYTHON" dycommu-tgat.py \
    --inputs democsv \
    --output-root example_outputs \
    --n-clusters 5 \
    --temporal-configs 25:3,50:3,100:3,200:3 \
    --no-resume

echo
echo "Summary:     example_outputs/summary.csv"
echo "Communities: example_outputs/node-time-communities.csv"
