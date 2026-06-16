#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Build a char-level vocabulary from corpus.txt.

Output: vocab.json mapping char -> id. Reserves ids for special tokens:
  <pad>=0, <bos>=1, <eos>=2, <unk>=3
then the top-N most frequent characters.

The vocab must cover the characters the IME can emit. Any char in a candidate
sentence at inference time that isn't in the vocab maps to <unk>; keep N large
enough (e.g. 8000) that common Traditional Chinese is fully covered.
"""

import argparse
import json
import sys
from collections import Counter

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--out", default="vocab.json")
    ap.add_argument("--top-n", type=int, default=8000,
                    help="number of most-frequent chars to keep")
    ap.add_argument("--min-count", type=int, default=5,
                    help="drop chars rarer than this even if within top-n")
    args = ap.parse_args()

    counter = Counter()
    n_lines = 0
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            counter.update(line.rstrip("\n"))
            n_lines += 1
            if n_lines % 500_000 == 0:
                print(f"[vocab] scanned {n_lines} lines, "
                      f"{len(counter)} distinct chars", file=sys.stderr)

    # Most common, filtered by min-count, capped at top-n.
    kept = [c for c, n in counter.most_common() if n >= args.min_count]
    kept = kept[: args.top_n]

    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for c in kept:
        if c not in vocab:
            vocab[c] = len(vocab)

    coverage = sum(counter[c] for c in kept)
    total = sum(counter.values())
    print(f"[vocab] {len(vocab)} tokens ({len(SPECIALS)} special + {len(kept)} "
          f"chars), char coverage {coverage / total:.4%}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)
    print(f"[vocab] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
