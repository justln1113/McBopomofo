#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Train a char-level LSTM next-char language model on corpus.txt.

The architecture is deliberately the simplest thing the C++ inference side can
faithfully reimplement by hand:

  embedding (V x E)
    -> single-layer LSTM (E -> H)   [standard 4-gate formulation]
    -> linear (H -> V)              [tied or untied; untied for simplicity]

At inference the C++ side runs one LSTM step per character and computes a
selective softmax over only the homophones at each position, so we keep one
layer and a modest hidden size. Bigger/deeper would complicate the hand port for
little gain on this task.
"""

import argparse
import json
import math
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # batch_first so shapes are (B, T, *). Single layer for a simple port.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                            batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        e = self.embed(x)
        out, hidden = self.lstm(e, hidden)
        logits = self.proj(out)
        return logits, hidden


class LineDataset(Dataset):
    """Each item is one corpus line encoded as ids with <bos>/<eos>, returned as
    (input, target) where target is input shifted by one."""

    def __init__(self, path: str, vocab: dict, max_len: int):
        self.bos = vocab["<bos>"]
        self.eos = vocab["<eos>"]
        self.unk = vocab["<unk>"]
        self.vocab = vocab
        self.max_len = max_len
        self.lines = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    self.lines.append(line)
        print(f"[train] loaded {len(self.lines)} lines", file=sys.stderr)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        line = self.lines[i][: self.max_len - 2]
        ids = [self.bos] + [self.vocab.get(c, self.unk) for c in line] + [self.eos]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate(batch, pad_id=0):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    maxlen = max(lengths)
    bx = torch.full((len(xs), maxlen), pad_id, dtype=torch.long)
    by = torch.full((len(ys), maxlen), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        bx[i, : len(x)] = x
        by[i, : len(y)] = y
    return bx, by


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    # Defaults are sized to stay within ~16GB. The per-step logits tensor is
    # batch * seq * vocab floats; with vocab ~6k, batch=128/max-len=128 needs
    # ~1.25GB transient *for logits alone*, which (compounded by the MPS pool
    # only ever growing) is enough to OOM a 16GB machine. Corpus lines average
    # ~29 chars, so max-len=96 truncates almost nothing.
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu",
                    help="cpu | mps | cuda. cpu is the safe default: stable and "
                         "memory-bounded. mps is faster but its memory pool only "
                         "grows; keep batch small and watch the first 200 steps.")
    # 0 = load in the main process. The dataset is already fully in RAM, so
    # workers add little; on macOS they use 'spawn', which copies all lines into
    # each worker -- avoid that memory multiplier by default.
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    device = torch.device(args.device)
    print(f"[train] device={device} vocab={vocab_size} "
          f"embed={args.embed_dim} hidden={args.hidden_dim}", file=sys.stderr)

    ds = LineDataset(args.corpus, vocab, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=args.num_workers,
                    drop_last=True)

    model = CharLSTM(vocab_size, args.embed_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # ignore_index=0 so padding does not contribute to the loss.
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {n_params/1e6:.2f}M params", file=sys.stderr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0
        t0 = time.time()
        for step, (bx, by) in enumerate(dl):
            bx, by = bx.to(device), by.to(device)
            logits, _ = model(bx)
            loss = loss_fn(logits.reshape(-1, vocab_size), by.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            seen += 1
            if (step + 1) % args.log_every == 0:
                avg = running / seen
                rate = seen * args.batch_size / (time.time() - t0)
                print(f"[train] epoch {epoch} step {step+1} "
                      f"loss {avg:.4f} ppl {math.exp(avg):.1f} "
                      f"({rate:.0f} seq/s)", file=sys.stderr)
                running = 0.0
                seen = 0
                t0 = time.time()
                # MPS never shrinks its allocation pool on its own; release the
                # cached blocks periodically so a long run does not creep up to
                # an OOM. Cheap and a no-op on other backends.
                if device.type == "mps":
                    torch.mps.empty_cache()

        # Save after each epoch (checkpoint), with everything export needs.
        torch.save({
            "state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
        }, args.out)
        print(f"[train] saved checkpoint after epoch {epoch} -> {args.out}",
              file=sys.stderr)

    print("[train] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
