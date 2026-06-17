#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Staged conversion-accuracy scorecard for the McBopomofo rescorer pipeline.

Runs a benchmark (gold<TAB>readings) through NBestGapHarness and reports a metric
for EACH pipeline stage, so a regression in any one stage is visible even when a
later stage masks it in the final number:

  plain-walk        CER / sentence-acc   -> guards LM data + unigram walk
  n-best oracle     CER + coverage       -> guards walkNBest (candidate generation)
  rescored          CER / sentence-acc   -> guards the neural rescorer
  lift = plain - rescored                -> the rescorer's contribution
  latency p50/p95 (us)                   -> guards inference perf

CER is per-character (Chinese standard; input readings ↔ output chars are 1:1, so
CER is positional mismatch). Invariant checked: oracle <= rescored <= plain.
Emits a scorecard JSON (timestamp, git commit, model+bench sha, all metrics) and
appends a compact line to scorecards/history.jsonl for eval_diff.py to compare.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict

from measure_headroom import load_lexicon

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, ".."))
ENGINE_BUILD = os.path.normpath(os.path.join(HERE, "..", "..", "Engine", "build"))


def cer(pred: str, gold: str) -> tuple[int, int]:
    """(errors, length). Positional when lengths match (the reachable case),
    else Levenshtein distance so a length mismatch still scores sanely."""
    if len(pred) == len(gold):
        return sum(p != g for p, g in zip(pred, gold)), len(gold)
    # Levenshtein fallback (rare: only if a reading was dropped).
    m, n = len(gold), len(pred)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (gold[i - 1] != pred[j - 1]))
        prev = cur
    return prev[n], max(m, 1)


def sha12(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            text=True).strip()
    except Exception:
        return "unknown"


def run_harness(harness, data, bench, weights, vocab, nbest):
    out = subprocess.run(
        [harness, data, bench, weights, vocab, str(nbest)],
        capture_output=True, text=True, check=True)
    return out.stdout


def micro(pairs):
    """pairs: list of (errors, length) -> (CER, sentence_accuracy)."""
    errs = sum(e for e, _ in pairs)
    chars = sum(l for _, l in pairs)
    sent_ok = sum(1 for e, _ in pairs if e == 0)
    return (errs / chars if chars else 0.0,
            sent_ok / len(pairs) if pairs else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True, help="gold<TAB>readings TSV")
    ap.add_argument("--label", required=True, help="e.g. golden / heldout")
    ap.add_argument("--data", default=os.path.join(DATA_DIR, "data.txt"))
    ap.add_argument("--weights",
                    default=os.path.join(DATA_DIR, "rescorer-weights.bin"))
    ap.add_argument("--vocab",
                    default=os.path.join(DATA_DIR, "rescorer-vocab.txt"))
    ap.add_argument("--harness",
                    default=os.path.join(ENGINE_BUILD, "NBestGapHarness"))
    ap.add_argument("--nbest", type=int, default=5)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "scorecards"))
    args = ap.parse_args()

    homo, _, char_reading = load_lexicon(args.data)

    def hardness(gold):  # max homophone-set size among the sentence's chars
        return max((len(homo.get(char_reading.get(c, ""), {})) for c in gold),
                   default=0)

    csv_text = run_harness(args.harness, args.data, args.bench, args.weights,
                           args.vocab, args.nbest)
    rows = defaultdict(list)
    for r in csv.DictReader(csv_text.splitlines()):
        rows[int(r["sid"])].append(r)

    plain, oracle, rescored = [], [], []
    cov = 0
    usecs = []
    by_len = defaultdict(lambda: [[], []])   # bucket -> [plain_pairs, resc_pairs]
    by_hard = {"ambiguous": [[], []], "easy": [[], []]}
    for cs in rows.values():
        gold = cs[0]["gold"]
        usecs.append(int(cs[0]["usec"]))
        pv = next(c["value"] for c in cs if c["is_plaintop"] == "1")
        rv = next(c["value"] for c in cs if c["is_pick"] == "1")
        ov = min((c["value"] for c in cs), key=lambda v: cer(v, gold)[0])
        p, o, rr = cer(pv, gold), cer(ov, gold), cer(rv, gold)
        plain.append(p); oracle.append(o); rescored.append(rr)
        if any(c["value"] == gold for c in cs):
            cov += 1
        b = min((len(gold) - 6) // 5, 3)  # 6-10,11-15,16-20,21+
        by_len[b][0].append(p); by_len[b][1].append(rr)
        k = "ambiguous" if hardness(gold) >= 15 else "easy"
        by_hard[k][0].append(p); by_hard[k][1].append(rr)

    n = len(rows)
    plain_cer, plain_sa = micro(plain)
    oracle_cer, oracle_sa = micro(oracle)
    resc_cer, resc_sa = micro(rescored)
    usecs.sort()
    def pctl(p):
        return usecs[min(len(usecs) - 1, int(p / 100 * len(usecs)))] if usecs else 0

    invariant_ok = oracle_cer <= resc_cer + 1e-9 <= plain_cer + 1e-9 or \
        (oracle_cer <= resc_cer + 1e-9 and resc_cer <= plain_cer + 1e-9)

    metrics = {
        "n_sentences": n,
        "coverage": round(cov / n, 4) if n else 0.0,
        "plain_cer": round(plain_cer, 4),
        "plain_sent_acc": round(plain_sa, 4),
        "oracle_cer": round(oracle_cer, 4),
        "rescored_cer": round(resc_cer, 4),
        "rescored_sent_acc": round(resc_sa, 4),
        "lift_cer": round(plain_cer - resc_cer, 4),
        "lift_sent_acc": round(resc_sa - plain_sa, 4),
        "latency_us_p50": pctl(50),
        "latency_us_p95": pctl(95),
        "by_len_rescored_cer": {
            ["6-10", "11-15", "16-20", "21+"][b]: round(micro(v[1])[0], 4)
            for b, v in sorted(by_len.items())},
        "by_hardness_rescored_cer": {
            k: round(micro(v[1])[0], 4) for k, v in by_hard.items() if v[1]},
        "invariant_oracle_le_rescored_le_plain": invariant_ok,
    }

    scorecard = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "git_commit": git_commit(),
        "model_sha": sha12(args.weights),
        "bench_sha": sha12(args.bench),
        "nbest": args.nbest,
        "metrics": metrics,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = scorecard["timestamp"].replace(":", "").replace("-", "")
    path = os.path.join(args.out_dir, f"{args.label}-{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "history.jsonl"), "a",
              encoding="utf-8") as f:
        f.write(json.dumps(scorecard, ensure_ascii=False) + "\n")

    print(f"\n=== scorecard [{args.label}] n={n} commit={scorecard['git_commit']} "
          f"model={scorecard['model_sha']} ===")
    print(f"  coverage (gold in n-best={args.nbest}) : {metrics['coverage']*100:5.1f}%")
    print(f"  plain-walk   CER {plain_cer*100:5.2f}%   sent-acc {plain_sa*100:5.1f}%")
    print(f"  n-best oracle CER {oracle_cer*100:5.2f}%   (re-rank ceiling)")
    print(f"  RESCORED     CER {resc_cer*100:5.2f}%   sent-acc {resc_sa*100:5.1f}%")
    print(f"  lift          CER {metrics['lift_cer']*100:+5.2f}pp  "
          f"sent-acc {metrics['lift_sent_acc']*100:+5.1f}pp")
    print(f"  latency/sent  p50 {pctl(50)}us  p95 {pctl(95)}us")
    print(f"  by length (rescored CER): {metrics['by_len_rescored_cer']}")
    print(f"  by hardness  (rescored CER): {metrics['by_hardness_rescored_cer']}")
    if not invariant_ok:
        print("  !! INVARIANT VIOLATED: expected oracle <= rescored <= plain",
              file=sys.stderr)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
