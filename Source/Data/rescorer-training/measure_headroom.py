#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Measure how much "headroom" a soft personalization bias would have.

The neural rescorer makes its final pick by argmax of

    combined = path.score(unigram) + lambda * modelScore   (+ mu * personalScore)

A personal-bias term `mu * personalScore` only "takes effect" if it can clear
the gap the engine already has between its top pick and the user's preferred
homophone. This script measures that gap on REAL sentences so we can size mu
from data instead of guessing, and decide whether the two-layer design is sound.

For each single-char homophone position in sampled corpus sentences we look at
the decision "actual char written by the corpus" vs "the unigram-top homophone":

  u = unigram_score(top) - unigram_score(actual)        >= 0  (unigram's pull)
  m = logP(original sentence) - logP(sentence w/ char swapped to top)
                                                          (the LSTM's pull, FULL
                                                           path re-score, so the
                                                           downstream signal the
                                                           rescorer relies on is
                                                           included)

The rescorer (lambda=1) keeps the actual char iff  m - u > 0.

Personalization headroom: when the user prefers a char the engine does NOT pick,
mu*personalScore must exceed the engine's net preference for its own pick. The
distributions of u, m, and the combined hurdle tell us that scale, and how often
decisions are near-ties (where a small soft bias is a free, safe tiebreaker).

Uses the SAME full-vocab log-softmax as the C++ LstmRescorerModel, so numbers
are faithful to production.
"""

import argparse
import json
import statistics
import sys

import torch
import torch.nn.functional as F

from train import CharLSTM
from evaluate import CASES, sentence_logprob

CJK = lambda ch: "一" <= ch <= "鿿"  # keep real Han chars only


def load_lexicon(path):
    """reading -> sorted [(char, score)] ; char -> dominant reading."""
    by_reading = {}
    char_best = {}  # char -> (score, reading) of its most probable reading
    with open(path, encoding="utf-8") as f:
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
            by_reading.setdefault(rd, []).append((val, sc))
            if val not in char_best or sc > char_best[val][0]:
                char_best[val] = (sc, rd)
    homo = {}          # reading -> {char: score}
    top = {}           # reading -> (char, score) unigram-best
    for rd, lst in by_reading.items():
        d = {}
        for v, s in lst:
            if v not in d or s > d[v]:
                d[v] = s
        homo[rd] = d
        top[rd] = max(d.items(), key=lambda kv: kv[1])
    char_reading = {c: r for c, (s, r) in char_best.items()}
    return homo, top, char_reading


def sample_sentences(path, n, lo, hi, stride):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            s = line.rstrip("\n")
            if lo <= len(s) <= hi and all(CJK(c) or c in "，。、" for c in s):
                out.append(s)
                if len(out) >= n:
                    break
    return out


@torch.no_grad()
def batch_total_logp(model, vocab, sents, device):
    """Total log P(s) for each sentence, batched."""
    bos, eos, unk = vocab["<bos>"], vocab["<eos>"], vocab["<unk>"]
    seqs = [[bos] + [vocab.get(c, unk) for c in s] + [eos] for s in sents]
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), L - 1, dtype=torch.long, device=device)
    y = torch.zeros(len(seqs), L - 1, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), L - 1, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        n = len(s) - 1
        x[i, :n] = torch.tensor(s[:-1], device=device)
        y[i, :n] = torch.tensor(s[1:], device=device)
        mask[i, :n] = True
    logits, _ = model(x)
    logp = F.log_softmax(logits, dim=-1)
    tok = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    tok = tok.masked_fill(~mask, 0.0)
    return tok.sum(dim=1).tolist()


def pct(xs, ps):
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(p / 100 * len(xs)))] for p in ps]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--data", default="data.txt")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sentences", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=37)  # spread the sample out
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    device = torch.device(args.device)
    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)
    ckpt = torch.load(args.model, map_location=device)
    model = CharLSTM(ckpt["vocab_size"], ckpt["embed_dim"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    homo, top, char_reading = load_lexicon(args.data)
    print(f"[lex] {len(homo)} readings, "
          f"{sum(len(v) for v in homo.values())} char entries", file=sys.stderr)

    sents = sample_sentences(args.corpus, args.sentences,
                             args.min_len, args.max_len, args.stride)
    print(f"[corpus] sampled {len(sents)} sentences", file=sys.stderr)

    # Build the work list: (sentence, position, actual_char, top_char, u).
    # Only positions where the unigram-top homophone differs from what was
    # actually written -- those are the decisions personalization would contest.
    work = []
    for s in sents:
        for i, c in enumerate(s):
            r = char_reading.get(c)
            if r is None:
                continue
            d = homo.get(r)
            if not d or len(d) < 2:
                continue
            tc, ts = top[r]
            if tc == c:
                continue  # unigram already agrees with the corpus; not contested
            u = ts - d[c]  # >= 0
            work.append((s, i, c, tc, u))

    print(f"[work] {len(work)} contested homophone positions", file=sys.stderr)

    # Score original vs swapped, batched. Build paired list.
    originals = [w[0] for w in work]
    swapped = [w[0][:w[1]] + w[3] + w[0][w[1] + 1:] for w in work]

    def all_logp(lst):
        out = []
        for k in range(0, len(lst), args.batch):
            out += batch_total_logp(model, vocab, lst[k:k + args.batch], device)
            print(f"\r  scored {min(k + args.batch, len(lst))}/{len(lst)}",
                  end="", file=sys.stderr)
        print(file=sys.stderr)
        return out

    # Cache identical originals to avoid rescoring the same sentence repeatedly.
    uniq = {}
    for s in originals:
        uniq[s] = None
    ukeys = list(uniq)
    uvals = all_logp(ukeys)
    for k, v in zip(ukeys, uvals):
        uniq[k] = v
    lp_orig = [uniq[s] for s in originals]
    lp_swap = all_logp(swapped)

    M, U, NET, HURDLE = [], [], [], []
    keep = flip = 0
    for (s, i, c, tc, u), lo_, ls_ in zip(work, lp_orig, lp_swap):
        m = lo_ - ls_           # model's pull toward the actual (correct) char
        net = m - u             # rescorer keeps actual iff net > 0
        M.append(m); U.append(u); NET.append(net)
        # If the user instead wanted the char the engine did NOT choose, the
        # personal term must clear the engine's margin over that char. Take the
        # engine's own decisive margin magnitude as the representative hurdle.
        HURDLE.append(abs(net))
        if net > 0: keep += 1
        else: flip += 1

    ps = [5, 25, 50, 75, 95]
    def row(name, xs):
        q = pct(xs, ps)
        print(f"{name:22s} mean {statistics.mean(xs):+7.3f} | "
              + " ".join(f"p{p}={v:+7.3f}" for p, v in zip(ps, q)))

    print("\n================ PERSONALIZATION HEADROOM (nats) ================")
    print(f"contested positions: {len(work)}   "
          f"(engine keeps corpus char: {keep}  flips to unigram-top: {flip})\n")
    row("model pull  m", M)
    row("unigram gap u", U)
    row("net = m - u", NET)
    row("|net| (mu hurdle)", HURDLE)
    near = sum(1 for x in NET if abs(x) < 1.0)
    print(f"\nnear-ties |net|<1.0 nat: {near}/{len(NET)} "
          f"({100*near/len(NET):.1f}%)  <- free tiebreaker zone for a soft bias")
    big = sum(1 for x in HURDLE if x > 5.0)
    print(f"hurdle >5 nats: {big}/{len(HURDLE)} ({100*big/len(HURDLE):.1f}%) "
          f"<- engine strongly committed; personalization should NOT override")

    print("\n================ CANONICAL CASES (anchor) ================")
    for correct, wrong in CASES:
        lc, _ = sentence_logprob(model, vocab, correct, device)
        lw, _ = sentence_logprob(model, vocab, wrong, device)
        print(f"  margin={lc-lw:+7.3f}  正:{correct}")


if __name__ == "__main__":
    main()
