#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Regression gate: compare the two most recent scorecards for a label and fail
(exit 1) if any stage regressed past its threshold.

This is what catches a single stage dropping even when the final number doesn't
move: every staged metric is checked independently, plus the monotonic invariant.
Run after eval_scorecard.py; wire into a pre-push hook for the fast golden bench.

  python eval_diff.py --label golden          # compare last two golden runs
  python eval_diff.py --label heldout --baseline <commit>   # pin a baseline
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (metric, direction, absolute-threshold). direction: lower=lower-is-better.
# A regression is a move in the bad direction by more than the threshold.
CHECKS = [
    ("plain_cer", "lower", 0.003),       # LM / walk regressed
    ("oracle_cer", "lower", 0.003),      # candidate generation (walkNBest) regressed
    ("coverage", "higher", 0.01),        # gold fell out of the n-best
    ("rescored_cer", "lower", 0.003),    # rescorer regressed
    ("rescored_sent_acc", "higher", 0.005),
]
LATENCY_REL = 0.20  # p95 latency regression if it grows > 20%


def load_history(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick(history, label, baseline_commit, baseline_file):
    entries = [e for e in history if e["label"] == label]
    if len(entries) < 1:
        return None, None
    cur = entries[-1]
    if baseline_file:  # pinned known-good scorecard (used by the pre-push gate)
        with open(baseline_file, encoding="utf-8") as f:
            base = json.load(f)
    elif baseline_commit:
        base = next((e for e in entries if e["git_commit"] == baseline_commit),
                    None)
    else:
        base = entries[-2] if len(entries) >= 2 else None
    return base, cur


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--baseline", default=None,
                    help="git commit to compare against (default: previous run)")
    ap.add_argument("--baseline-file", default=None,
                    help="pinned known-good scorecard JSON (used by the gate)")
    ap.add_argument("--history",
                    default=os.path.join(HERE, "scorecards", "history.jsonl"))
    ap.add_argument("--strict-golden", action="store_true", default=True,
                    help="for the golden label, any sent-acc drop fails")
    args = ap.parse_args()

    base, cur = pick(load_history(args.history), args.label, args.baseline,
                     args.baseline_file)
    if cur is None:
        print(f"[diff] no scorecards for label '{args.label}'", file=sys.stderr)
        return 1
    cm, regressions = cur["metrics"], []

    if not cm.get("invariant_oracle_le_rescored_le_plain", True):
        regressions.append("INVARIANT oracle<=rescored<=plain violated")

    if base is None:
        print(f"[diff] {args.label}: first run (commit {cur['git_commit']}), "
              f"nothing to compare; recording baseline.")
        _summary(cur)
        return 0

    bm = base["metrics"]
    print(f"[diff] {args.label}: {base['git_commit']}({base['model_sha']}) "
          f"-> {cur['git_commit']}({cur['model_sha']})")

    golden = args.label == "golden"
    for metric, direction, thr in CHECKS:
        if metric not in cm or metric not in bm:
            continue
        old, new = bm[metric], cm[metric]
        if golden and args.strict_golden and metric == "rescored_sent_acc":
            thr = 0.0  # any drop on the hard gate is a regression
        bad = (new - old > thr) if direction == "lower" else (old - new > thr)
        arrow = "↑" if new > old else ("↓" if new < old else "=")
        flag = "  REGRESSION" if bad else ""
        print(f"    {metric:22s} {old:.4f} -> {new:.4f} {arrow}{flag}")
        if bad:
            regressions.append(f"{metric} {old:.4f}->{new:.4f}")

    # Latency is informational, NOT a gate: single-sample wall-clock on a shared
    # dev machine swings far too much run-to-run (we've seen +60% from pure load)
    # to gate a push on. We print the delta so a real, sustained perf regression
    # is visible in the trend; the hard gate is accuracy + the invariant.
    op, np_ = bm.get("latency_us_p95", 0), cm.get("latency_us_p95", 0)
    note = " (noisy; informational)" if op and np_ / op - 1 > LATENCY_REL else ""
    print(f"    latency_us_p95         {op} -> {np_}{note}")

    if regressions:
        print(f"\n[diff] FAIL — {len(regressions)} regression(s):", file=sys.stderr)
        for r in regressions:
            print(f"    - {r}", file=sys.stderr)
        return 1
    print(f"\n[diff] OK — no regression for '{args.label}'")
    return 0


def _summary(cur):
    m = cur["metrics"]
    print(f"    coverage={m['coverage']} plain_cer={m['plain_cer']} "
          f"oracle_cer={m['oracle_cer']} rescored_cer={m['rescored_cer']} "
          f"rescored_sent_acc={m['rescored_sent_acc']}")


if __name__ == "__main__":
    sys.exit(main())
