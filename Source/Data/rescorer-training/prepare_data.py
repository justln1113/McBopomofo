#!/usr/bin/env python3
# Copyright (c) 2025 and onwards The McBopomofo Authors.
# Licensed under the MIT-style license of the McBopomofo project.
"""Download, clean, and normalize Taiwan Traditional Chinese corpus.

Output: corpus.txt, one cleaned sentence/line, ready for char-level LM training.

The model scores candidate sentences that the IME would emit, so normalization
here must match IME output conventions: Traditional Chinese characters, standard
punctuation, no HTML, no stray Latin/markup noise. We segment into sentence-ish
lines at Chinese sentence punctuation so the LSTM learns sentence-scale fluency
rather than document drift.
"""

import argparse
import sys
import unicodedata

import regex as re

# Datasets to pull, in priority order. Personal/research use -> license
# unrestricted; these are the cleanest native Taiwan Traditional Chinese sources.
# Each entry: (hf_dataset, config, split, text_field).
DATASETS = [
    ("zetavg/zh-tw-wikipedia", None, "train", "html"),  # clean 台灣正體 wiki
    ("liswei/Taiwan-Text-Excellence-2B", None, "train", "text"),  # multi-domain
]

# Characters we keep: CJK unified ideographs (+ ext A), plus the punctuation the
# IME actually produces. Everything else (Latin, emoji, control, markup) is a
# signal to split/drop, since the model scores Chinese fluency.
CJK = r"\p{Han}"
# Standard Chinese punctuation McBopomofo emits.
KEEP_PUNCT = "，。、；：？！「」『』（）《》〈〉…—～·．"
SENTENCE_END = "。？！…"

_html_tag = re.compile(r"<[^>]+>")
_html_entity = re.compile(r"&[a-zA-Z#0-9]+;")
_md_link = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) -> text
_md_marks = re.compile(r"[*_`#>|]+")
_ws = re.compile(r"\s+")
# A run of allowed chars (Han + kept punctuation). Anything else is a boundary.
_keep_run = re.compile(rf"[{CJK}{re.escape(KEEP_PUNCT)}]+")


def clean_text(raw: str) -> str:
    """Strip markup and normalize a raw document to kept characters/punct."""
    s = unicodedata.normalize("NFKC", raw)
    s = _md_link.sub(r"\1", s)
    s = _html_tag.sub(" ", s)
    s = _html_entity.sub(" ", s)
    s = _md_marks.sub(" ", s)
    # NFKC turns fullwidth punct into halfwidth/ASCII for some marks; re-map the
    # few ASCII ones the IME would emit as fullwidth back to fullwidth.
    s = (s.replace(",", "，").replace("?", "？").replace("!", "！")
          .replace(";", "；").replace(":", "："))
    return s


def split_sentences(cleaned: str):
    """Yield sentence-ish lines: runs of kept chars, broken at sentence-end
    punctuation. Drops lines too short or with too little Han content."""
    for run in _keep_run.findall(cleaned):
        # Break the run into sentences at sentence-ending punctuation, keeping
        # the punctuation attached.
        buf = []
        for ch in run:
            buf.append(ch)
            if ch in SENTENCE_END:
                line = "".join(buf).strip()
                buf = []
                if _is_good_line(line):
                    yield line
        tail = "".join(buf).strip()
        if _is_good_line(tail):
            yield tail


def _is_good_line(line: str) -> bool:
    if len(line) < 6:
        return False
    han = sum(1 for c in line if re.match(CJK, c))
    # Require the line to be mostly Han (filters punctuation-only / noise lines).
    return han >= 5 and han / len(line) >= 0.6


def iter_corpus(max_chars: int, limit_docs: int | None):
    """Stream documents from the datasets, cleaning and splitting, until
    max_chars characters of output have been produced."""
    from datasets import load_dataset

    produced = 0
    for name, config, split, field in DATASETS:
        print(f"[prepare] streaming {name} (field={field})", file=sys.stderr)
        try:
            ds = load_dataset(name, config, split=split, streaming=True)
        except Exception as e:  # noqa: BLE001
            print(f"[prepare] WARN could not load {name}: {e}", file=sys.stderr)
            continue

        docs = 0
        for row in ds:
            raw = row.get(field) or row.get("text") or ""
            if not raw:
                continue
            cleaned = clean_text(raw)
            for line in split_sentences(cleaned):
                yield line
                produced += len(line) + 1
                if produced >= max_chars:
                    print(f"[prepare] reached max-chars={max_chars}",
                          file=sys.stderr)
                    return
            docs += 1
            if limit_docs and docs >= limit_docs:
                break
        print(f"[prepare] {name}: {docs} docs, {produced} chars so far",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="corpus.txt")
    ap.add_argument("--max-chars", type=lambda s: int(s.replace("_", "")),
                    default=50_000_000,
                    help="stop after this many output characters")
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="optional cap on docs per dataset (for quick tests)")
    args = ap.parse_args()

    n_lines = 0
    n_chars = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for line in iter_corpus(args.max_chars, args.limit_docs):
            f.write(line)
            f.write("\n")
            n_lines += 1
            n_chars += len(line) + 1
            if n_lines % 100_000 == 0:
                print(f"[prepare] {n_lines} lines, {n_chars} chars",
                      file=sys.stderr)

    print(f"[prepare] DONE: {n_lines} lines, {n_chars} chars -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
