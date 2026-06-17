#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Build a held-out Traditional Chinese eval set with ZERO training overlap.

The rescorer LM was trained on corpus.txt. If the benchmark overlaps training,
rescorer accuracy is inflated by memorization. So we stream FRESH text from the
same Taiwan datasets and keep only pure-Han spans that do NOT appear in training.

Leakage subtlety: corpus.txt lines carry punctuation, but the bench wants pure-Han
spans (the annotator/IME work on Han only). An exact-LINE dedup would miss a span
that is a corpus line minus its punctuation. So we dedup at the level of *maximal
Han runs*: extract maximal Han runs from corpus.txt into a seen-set, and drop any
held-out Han run already in it. Residual risk (a held-out run that is a proper
substring of a longer training run in a different context) is accepted and noted.

Output: one pure-Han held-out sentence per line (6-24 chars), ready for
gen_bench_readings.py.
"""

import argparse
import sys

import regex as re

from prepare_data import clean_text, iter_corpus

_han_run = re.compile(r"\p{Han}+")


def load_seen_runs(corpus_path):
    """Hashes of every maximal Han run in the training corpus (any length)."""
    seen = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            for run in _han_run.findall(line):
                seen.add(hash(run))
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus.txt",
                    help="training corpus to exclude overlap against")
    ap.add_argument("--out", default="heldout.txt")
    ap.add_argument("--target", type=int, default=5000,
                    help="number of held-out sentences to collect")
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--max-docs", type=int, default=400_000,
                    help="safety cap on docs scanned per dataset")
    args = ap.parse_args()

    print(f"[heldout] loading training Han runs from {args.corpus} ...",
          file=sys.stderr)
    seen = load_seen_runs(args.corpus)
    print(f"[heldout] {len(seen)} distinct training Han runs", file=sys.stderr)

    collected = set()
    scanned = 0
    overlap = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for doc_line in iter_corpus(args.max_docs):
            scanned += 1
            for run in _han_run.findall(clean_text(doc_line)):
                if not (args.min_len <= len(run) <= args.max_len):
                    continue
                h = hash(run)
                if h in seen:
                    overlap += 1
                    continue
                if run in collected:
                    continue
                collected.add(run)
                out.write(run + "\n")
                if len(collected) >= args.target:
                    print(f"\n[heldout] DONE: {len(collected)} sentences "
                          f"({overlap} training-overlap spans skipped)",
                          file=sys.stderr)
                    return
            if scanned % 5000 == 0:
                print(f"\r  scanned {scanned} lines, collected {len(collected)}",
                      end="", file=sys.stderr)
    print(f"\n[heldout] stream exhausted: {len(collected)} sentences "
          f"({overlap} overlap skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
