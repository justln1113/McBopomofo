#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Score competing homophone sentences under the trained model.

This is the task-level sanity check that matters more than perplexity: for each
(correct, wrong) pair the rescorer must eventually disambiguate, does the model
assign the correct sentence a higher log-probability? The candidates differ only
at the homophone span and have equal length, so a plain total log-prob
comparison is fair.

If the model prefers the correct sentence here, the C++ LstmRescorerModel (a
faithful port of this same forward pass) will flip the walk's choice to it.
"""

import argparse
import json
import math
import sys

import torch

from train import CharLSTM  # reuse the exact model definition

# (correct, wrong). The first two are the canonical cases from the project notes;
# the rest are clear fluency contrasts to gauge general discrimination.
CASES = [
    ("依你觀測的結果來辦", "一你觀測的結果來辦"),   # 依/一 ，近距離 bigram
    ("一首遺憾的歌", "一手遺憾的歌"),               # 首/手 ，線索「歌」在 3 字外
    ("我已經吃過晚餐了", "我以經吃過晚餐了"),       # 已/以
    ("現在幾點了", "現再幾點了"),                   # 在/再
    ("這是他的決定", "這是他的決訂"),               # 定/訂
]


def load(model_path: str, vocab_path: str, device: torch.device):
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    ckpt = torch.load(model_path, map_location=device)
    model = CharLSTM(ckpt["vocab_size"], ckpt["embed_dim"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, vocab


@torch.no_grad()
def sentence_logprob(model, vocab, s: str, device: torch.device):
    """Total log P(s) and per-char perplexity under the next-char LM."""
    bos, eos, unk = vocab["<bos>"], vocab["<eos>"], vocab["<unk>"]
    ids = [bos] + [vocab.get(c, unk) for c in s] + [eos]
    x = torch.tensor(ids[:-1], device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], device=device)
    logits, _ = model(x)
    logp = torch.log_softmax(logits[0], dim=-1)
    tok_lp = logp[range(y.shape[0]), y]
    total = tok_lp.sum().item()
    ppl = math.exp(-total / y.shape[0])
    return total, ppl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, vocab = load(args.model, args.vocab, device)

    n_correct = 0
    for correct, wrong in CASES:
        lc, pc = sentence_logprob(model, vocab, correct, device)
        lw, pw = sentence_logprob(model, vocab, wrong, device)
        win = lc > lw
        n_correct += win
        mark = "OK  " if win else "MISS"
        print(f"[{mark}] 正:{correct}  (logP={lc:8.3f} ppl={pc:6.2f})")
        print(f"       誤:{wrong}  (logP={lw:8.3f} ppl={pw:6.2f})"
              f"   margin={lc - lw:+.3f}")
    print(f"\n{n_correct}/{len(CASES)} 正確句勝出", file=sys.stderr)


if __name__ == "__main__":
    main()
