# Rescorer LSTM Training

Trains the char-level LSTM fluency scorer used by the second-pass neural
rescorer (see `Source/Engine/NeuralRescorer.h`). The model is a pure
**sentence-fluency** language model: it scores how natural a candidate Chinese
sentence reads. It never sees Bopomofo readings -- the lattice already
constrains which homophones are possible, so the model's only job is to prefer
依你 over 一你 by overall fluency.

This is a standalone Python project, separate from the C++ engine. The engine
side (walkNBest + NeuralRescorer scaffold) is already done and tested.

## Pipeline

```
prepare_data.py   download + clean + normalize corpus -> corpus.txt
build_vocab.py    char frequency -> vocab.json (top-N chars + specials)
train.py          char LSTM, next-char LM (PyTorch) -> model.pt
export_weights.py model.pt -> weights.bin (flat, memory-mappable by C++)
```

## Corpus (personal/research use, license unrestricted)

- Base: `zetavg/zh-tw-wikipedia` (clean 台灣正體; raw zhwiki dumps are
  simp/trad MIXED -- avoid those)
- Augment: `liswei/Taiwan-Text-Excellence-2B` (multi-domain Taiwan text)

AVOID (simp->trad pollution): uonlp/CulturaX, erhwenkuo/wikipedia-zhtw,
c4-zhtw, OSCAR, CC-100.

## Why CPU is (probably) enough

The model is tiny (1-3M params). The cost is data volume, and an LSTM can't
parallelize across timesteps. A fluency scorer saturates around a few hundred
million chars, so we subsample. ~50M chars trains on CPU in a few hours; ~300M
overnight. `device="mps"` may speed it up but PyTorch's LSTM on MPS can be
flaky -- fall back to CPU subsampling if so.

## Setup (uv)

```bash
cd Source/Data/rescorer-training
uv sync          # creates .venv and installs deps from pyproject.toml
```

## Run (uv)

```bash
uv run python prepare_data.py --max-chars 50_000_000   # prototype size
uv run python build_vocab.py
uv run python train.py --device cpu                     # or --device mps
uv run python export_weights.py --int8                  # quantized weights
```

Each step writes a file the next consumes: corpus.txt -> vocab.json ->
model.pt -> weights.bin (+ vocab.txt). All are gitignored.
