#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Annotate Han sentences with context-aware bopomofo readings for the IME bench.

The C++ engine only has reading->value lookup, so to drive the real walk we must
supply the correct per-character readings. We use g2pW (context-aware Mandarin
polyphone disambiguation, native zhuyin output) but as a SELECTOR over the IME's
OWN reading inventory rather than as the raw reading source:

  for each char: candidate readings = the char's readings in data.txt;
                 pick the one whose spelling matches g2pW's contextual prediction;
                 fall back to the highest-unigram (dominant) reading if no match.

Grounding readings in data.txt guarantees correct spelling (data.txt uses tone
MARKS ˊˇˋ˙ and unmarked 1st tone; g2pW emits tone NUMBERS) and gives Taiwan
readings for free (data.txt is the Taiwan IME's own LM). g2pW's training data is
Simplified CPP, so the data.txt grounding + fallback bounds its blast radius;
disagreement/fallback rates are reported so the noise floor is visible.

Input:  one Han sentence per line (lines with a TAB use the part before it as the
        sentence, so a gold\\t... file round-trips).
Output: <gold Han>\\t<reading1> <reading2> ...   (only fully-reachable sentences)
"""

import argparse
import sys
from collections import defaultdict

CJK = lambda ch: "一" <= ch <= "鿿"

# g2pW tone digit -> McBopomofo tone-mark spelling (1st tone unmarked, 5=neutral).
TONE = {"1": "", "2": "ˊ", "3": "ˇ", "4": "ˋ", "5": "˙"}


def g2pw_to_reading(z: str) -> str:
    """'ㄕㄤ4' -> 'ㄕㄤˋ' ; 'ㄉㄜ5' -> 'ㄉㄜ˙' ; 'ㄧ1' -> 'ㄧ'."""
    if not z or z[-1] not in TONE:
        return z or ""
    return z[:-1] + TONE[z[-1]]


def load_char_readings(data_path):
    """char -> {reading: score}, and char -> dominant reading (max score)."""
    by_char = defaultdict(dict)
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split(" ")
            if len(p) != 3:
                continue
            rd, val, sc = p
            if rd.startswith("_") or "-" in rd or len(val) != 1 or not CJK(val):
                continue
            sc = float(sc)
            if rd not in by_char[val] or sc > by_char[val][rd]:
                by_char[val][rd] = sc
    dominant = {c: max(rs, key=rs.get) for c, rs in by_char.items()}
    return by_char, dominant


def make_converter(model_dir):
    from g2pw import G2PWConverter
    conv = G2PWConverter(model_dir=model_dir, style="bopomofo", turnoff_tqdm=True)
    # The constructor does `num_workers if num_workers else config.num_workers`,
    # so passing 0 is ignored (0 is falsy) and the >0 config default spawns
    # workers that break under spawn. Override the attribute directly.
    conv.num_workers = 0
    return conv


def read_sentences(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if "\t" in s:
                s = s.split("\t", 1)[0]
            if s and all(CJK(c) for c in s):
                out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="one Han sentence per line")
    ap.add_argument("--data", default="../data.txt")
    ap.add_argument("--out", default="bench_readings.tsv")
    ap.add_argument("--model-dir", default="G2PWModel")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    by_char, dominant = load_char_readings(args.data)
    print(f"[lex] {len(by_char)} chars with single-syllable readings",
          file=sys.stderr)

    sents = read_sentences(args.input)
    print(f"[in] {len(sents)} candidate sentences", file=sys.stderr)

    conv = make_converter(args.model_dir)

    kept = 0
    n_chars = 0
    n_picked_g2pw = 0      # g2pW reading existed in data.txt and was used
    n_picked_nondom = 0    # ... and it differed from the dominant reading
    n_fallback = 0         # g2pW reading absent in data.txt -> dominant
    with open(args.out, "w", encoding="utf-8") as out:
        for k in range(0, len(sents), args.batch):
            chunk = sents[k:k + args.batch]
            preds = conv(chunk)
            for s, zhuyin in zip(chunk, preds):
                if len(zhuyin) != len(s):
                    continue  # alignment broke (shouldn't on all-CJK input)
                readings = []
                reachable = True
                for ch, z in zip(s, zhuyin):
                    rs = by_char.get(ch)
                    if not rs:
                        reachable = False
                        break
                    cand = g2pw_to_reading(z) if z else ""
                    if cand in rs:
                        readings.append(cand)
                        n_picked_g2pw += 1
                        if cand != dominant[ch]:
                            n_picked_nondom += 1
                    else:
                        readings.append(dominant[ch])
                        n_fallback += 1
                    n_chars += 1
                if reachable:
                    out.write(s + "\t" + " ".join(readings) + "\n")
                    kept += 1
            print(f"\r  annotated {min(k + args.batch, len(sents))}/{len(sents)}",
                  end="", file=sys.stderr)
    print(file=sys.stderr)
    print(f"[out] kept {kept}/{len(sents)} fully-reachable sentences -> {args.out}",
          file=sys.stderr)
    if n_chars:
        print(f"[stats] chars={n_chars}  g2pW-matched-data.txt={100*n_picked_g2pw/n_chars:.1f}%"
              f"  (of which non-dominant pick={100*n_picked_nondom/max(1,n_picked_g2pw):.1f}%)"
              f"  fallback-to-dominant={100*n_fallback/n_chars:.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
