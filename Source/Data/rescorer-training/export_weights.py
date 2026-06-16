#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Export a trained model.pt to a flat binary the C++ inference side reads.

The C++ RescorerModel reimplements the LSTM forward pass by hand and
memory-maps this file (same spirit as ParselessPhraseDB). The format is a small
header followed by raw float32 (or int8 if quantized) tensors in a fixed order,
so the C++ side needs no parser -- just offsets.

Layout (all little-endian):
  magic     : 8 bytes  "MCBLSTM1"
  dtype     : uint32   0 = float32, 1 = int8 (per-tensor scale follows)
  vocab     : uint32
  embed_dim : uint32
  hidden    : uint32
  reserved  : uint32   (0)
  then tensors back-to-back in this order:
    embed.weight        [vocab, embed]
    lstm W_ih           [4*hidden, embed]   (PyTorch gate order: i, f, g, o)
    lstm W_hh           [4*hidden, hidden]
    lstm b_ih           [4*hidden]
    lstm b_hh           [4*hidden]
    proj.weight         [vocab, hidden]
    proj.bias           [vocab]
  for int8 dtype, each tensor is preceded by a float32 scale; values are
  round(x/scale) clamped to [-127,127], dequantized as q*scale in C++.

Gate order note: PyTorch packs LSTM gates as [input, forget, cell, output].
The C++ side MUST use the same order. This is documented in the C++ loader.
"""

import argparse
import json
import struct
import sys

import numpy as np
import torch

MAGIC = b"MCBLSTM1"
TENSOR_ORDER = [
    "embed.weight",
    "lstm.weight_ih_l0",
    "lstm.weight_hh_l0",
    "lstm.bias_ih_l0",
    "lstm.bias_hh_l0",
    "proj.weight",
    "proj.bias",
]


def write_tensor_f32(f, arr: np.ndarray):
    f.write(arr.astype("<f4").tobytes())


def write_tensor_int8(f, arr: np.ndarray):
    # Symmetric per-tensor quantization.
    amax = float(np.abs(arr).max()) or 1.0
    scale = amax / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    f.write(struct.pack("<f", scale))
    f.write(q.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--out", default="weights.bin")
    ap.add_argument("--int8", action="store_true",
                    help="export int8-quantized weights (smaller file)")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    sd = ckpt["state_dict"]
    vocab_size = ckpt["vocab_size"]
    embed_dim = ckpt["embed_dim"]
    hidden = ckpt["hidden_dim"]

    # Sanity: confirm all expected tensors exist with expected shapes.
    expected = {
        "embed.weight": (vocab_size, embed_dim),
        "lstm.weight_ih_l0": (4 * hidden, embed_dim),
        "lstm.weight_hh_l0": (4 * hidden, hidden),
        "lstm.bias_ih_l0": (4 * hidden,),
        "lstm.bias_hh_l0": (4 * hidden,),
        "proj.weight": (vocab_size, hidden),
        "proj.bias": (vocab_size,),
    }
    for name, shape in expected.items():
        if name not in sd:
            print(f"[export] ERROR missing tensor {name}", file=sys.stderr)
            sys.exit(1)
        if tuple(sd[name].shape) != shape:
            print(f"[export] ERROR {name} shape {tuple(sd[name].shape)} "
                  f"!= expected {shape}", file=sys.stderr)
            sys.exit(1)

    dtype = 1 if args.int8 else 0
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIIII", dtype, vocab_size, embed_dim, hidden, 0))
        for name in TENSOR_ORDER:
            arr = sd[name].cpu().numpy()
            if args.int8:
                write_tensor_int8(f, arr)
            else:
                write_tensor_f32(f, arr)

    # Also emit the vocab as a parallel file the C++ side loads for char->id.
    with open(args.vocab, encoding="utf-8") as vf:
        vocab = json.load(vf)
    with open("vocab.txt", "w", encoding="utf-8") as out:
        # id-ordered, one token per line, so C++ reads it as a vector.
        for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
            out.write(tok.replace("\n", "\\n"))
            out.write("\n")

    n_params = sum(int(np.prod(s)) for s in expected.values())
    kind = "int8" if args.int8 else "float32"
    print(f"[export] wrote {args.out} ({kind}, ~{n_params/1e6:.2f}M params) "
          f"and vocab.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
