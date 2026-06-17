#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Turn real corpus sentences into bopomofo reading sequences for the C++ harness.

The C++ engine has only reading->value (forward) lookup, so to drive the real
grid/walk we must supply readings. We invert data.txt to char->dominant-reading
and emit, per line:   <gold Han sentence>\t<reading1> <reading2> ...

Each char's reading is the one under which that char has the highest unigram
score (its most probable pronunciation). By construction the gold char is always
a candidate for its assigned reading, so the gold path is always reachable by the
walk -- the harness can then measure how the rescorer ranks it among the n-best.
Sentences containing any char without a single-syllable reading are skipped so the
reading sequence stays aligned with the gold text.
"""

import argparse
import sys

from measure_headroom import load_lexicon, CJK


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="../data.txt")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--out", default="harness_readings.tsv")
    ap.add_argument("--sentences", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=37)
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=24)
    args = ap.parse_args()

    _, _, char_reading = load_lexicon(args.data)
    print(f"[lex] {len(char_reading)} chars with a single-syllable reading",
          file=sys.stderr)

    written = 0
    with open(args.corpus, encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if i % args.stride:
                continue
            s = line.rstrip("\n")
            if not (args.min_len <= len(s) <= args.max_len):
                continue
            if not all(CJK(c) for c in s):
                continue
            readings = [char_reading.get(c) for c in s]
            if any(r is None for r in readings):
                continue
            out.write(s + "\t" + " ".join(readings) + "\n")
            written += 1
            if written >= args.sentences:
                break
    print(f"[out] wrote {written} reading sequences -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
