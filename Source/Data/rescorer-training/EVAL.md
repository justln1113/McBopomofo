# Conversion-accuracy eval (staged CER regression tracking)

Automated, per-stage accuracy tracking for the McBopomofo conversion pipeline, so
a regression in any one stage is observable even when a later stage masks it in
the final number. Metric is **CER** (Character Error Rate; Chinese standard —
input readings ↔ output chars are 1:1, so CER is positional mismatch) plus
sentence accuracy.

## Stages tracked

| Metric | Guards |
| --- | --- |
| `plain_cer` / `plain_sent_acc` | LM data + unigram walk |
| `oracle_cer` + `coverage` (gold in n-best) | `walkNBest` candidate generation |
| `rescored_cer` / `rescored_sent_acc` | the neural rescorer |
| `lift = plain − rescored` | the rescorer's contribution |
| `latency_us_p50/p95` | inference perf |

Invariant checked every run: `oracle_cer ≤ rescored_cer ≤ plain_cer`. Metrics are
also stratified by sentence length and by homophone-ambiguity, so a regression
concentrated in hard cases isn't washed out by easy ones.

## Two bench tiers

- **Tier-1 golden** (`bench/golden.tsv`, committed, frozen): hand-curated hard
  homophone / polyphone / long-range cases, readings verified. Zero G2P noise;
  the pre-push hard gate. Source sentences in `bench/golden_src.txt`.
- **Tier-2 held-out** (generated, gitignored): ~5k sentences streamed fresh from
  the Taiwan corpus and **deduped against `corpus.txt`** (no training overlap, or
  rescorer numbers inflate). Readings annotated by g2pW. Wiki-only at present
  (`liswei/Taiwan-Text-Excellence-2B` is gated) — expect some 正/簡 orthography
  noise; it's a constant offset and does not affect regression deltas.

## Readings = g2pW as a selector over data.txt

The IME needs correct per-char bopomofo. `gen_bench_readings.py` runs g2pW
(context-aware polyphone disambiguation, native zhuyin) but uses it to **pick
among the char's own data.txt readings** (fallback: dominant). This grounds
spelling in the IME's inventory and yields Taiwan readings (和→ㄏㄢˋ) despite
g2pW's Simplified training data. Note: g2pW emits tone-sandhi readings (一→ㄧˋ)
where users type the citation tone (一→ㄧ); harmless here because the char stays
reachable and dominant.

## Run

```bash
# build the harness once
cmake -B ../../Engine/build -S ../../Engine -DCMAKE_BUILD_TYPE=Release
cmake --build ../../Engine/build --target NBestGapHarness

# (re)generate the held-out tier (network; uses g2pW, downloads G2PWModel once)
.venv/bin/python gen_heldout.py --target 5000 --out heldout.txt
.venv/bin/python gen_bench_readings.py --input heldout.txt --out bench_readings.tsv

# full eval (both tiers + regression diff vs pinned baselines)
./eval.sh
```

Each run writes `scorecards/<label>-<ts>.json` and appends to
`scorecards/history.jsonl` (both local/gitignored). The pinned known-good
`scorecards/baseline-*.json` are committed; `eval_diff.py --baseline-file`
compares against them.

## Pre-push gate

`hooks/pre-push` runs the fast golden bench and blocks the push on any regression.
Install locally:

```bash
ln -sf ../../Source/Data/rescorer-training/hooks/pre-push \
       "$(git rev-parse --git-dir)/hooks/pre-push"
```

It skips gracefully if the venv/harness/baseline aren't present, and is overridable
with `git push --no-verify`. After an intended accuracy change, refresh the pinned
baseline by copying the new `scorecards/<label>-*.json` over `baseline-<label>.json`.
