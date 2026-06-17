#!/usr/bin/env bash
# Full manual conversion-accuracy eval: staged scorecards for both bench tiers,
# then a regression diff against the pinned baselines. Run on every model retrain
# / engine change. Exits non-zero if any stage regressed.
#
# Tier-2 (held-out) needs bench_readings.tsv, which is generated (gitignored):
#   .venv/bin/python gen_heldout.py --target 5000 --out heldout.txt
#   .venv/bin/python gen_bench_readings.py --input heldout.txt --out bench_readings.tsv
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
HARNESS=../../Engine/build/NBestGapHarness
status=0

if [ ! -x "$HARNESS" ]; then
  echo "build the harness first: cmake --build ../../Engine/build --target NBestGapHarness"
  exit 2
fi

echo "### Tier-1 golden (frozen fixture)"
$PY eval_scorecard.py --bench bench/golden.tsv --label golden
$PY eval_diff.py --label golden --baseline-file scorecards/baseline-golden.json || status=1

if [ -f bench_readings.tsv ]; then
  echo; echo "### Tier-2 held-out"
  $PY eval_scorecard.py --bench bench_readings.tsv --label heldout
  $PY eval_diff.py --label heldout --baseline-file scorecards/baseline-heldout.json || status=1
else
  echo; echo "### Tier-2 held-out: bench_readings.tsv missing (see header to regenerate); skipped"
fi

[ $status -eq 0 ] && echo && echo "ALL OK" || { echo; echo "REGRESSION(S) DETECTED"; }
exit $status
