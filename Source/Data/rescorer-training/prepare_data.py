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

# Datasets to pull, interleaved round-robin (see iter_corpus) so the output is a
# mix rather than 100% of whichever comes first. Personal/research use -> license
# unrestricted; these are the cleanest native Taiwan Traditional Chinese sources.
# Each entry: (hf_dataset, config, split, text_field).
DATASETS = [
    ("zetavg/zh-tw-wikipedia", None, "train", "html"),  # clean 台灣正體 wiki
    ("liswei/Taiwan-Text-Excellence-2B", None, "train", "text"),  # news/科技/公視/wiki
]

# Characters we keep: CJK unified ideographs (+ ext A), Arabic digits, plus the
# punctuation the IME actually produces. Everything else (Latin, emoji, control,
# markup) is a signal to split/drop, since the model scores Chinese fluency.
CJK = r"\p{Han}"
# Keep digits: the IME emits them (「2024年」「第3名」), and dropping them split
# 「1956年10月29日，…」 at every digit, orphaning the unit char (年/月/日) onto the
# next line — ~25% of wiki lines started with such residue. NFKC has already
# folded fullwidth ０-９ to these ASCII digits, so this covers both forms.
DIGITS = "0-9"
# Standard Chinese punctuation McBopomofo emits.
KEEP_PUNCT = "，。、；：？！「」『』（）《》〈〉…—～·．"
SENTENCE_END = "。？！…"

_html_tag = re.compile(r"<[^>]+>")
_html_entity = re.compile(r"&[a-zA-Z#0-9]+;")
_md_link = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) -> text
_md_marks = re.compile(r"[*_`#>|]+")
_ws = re.compile(r"\s+")
# A run of allowed chars (Han + digits + kept punctuation). Anything else is a
# boundary.
_keep_run = re.compile(rf"[{CJK}{DIGITS}{re.escape(KEEP_PUNCT)}]+")


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


# Closing brackets/quotes that should stay attached to the sentence they end,
# rather than starting the next line.
_CLOSERS = "」』）》〉"


def split_sentences(cleaned: str):
    """Yield sentence-ish lines: runs of kept chars, broken at sentence-end
    punctuation. A closing quote/bracket immediately after the sentence-end
    punctuation is absorbed into the same line. Fragments without sentence-end
    punctuation (mostly wiki structural noise: titles, list items, table cells)
    are held to a stricter bar in _is_good_line."""
    for run in _keep_run.findall(cleaned):
        buf = []
        i = 0
        n = len(run)
        while i < n:
            ch = run[i]
            buf.append(ch)
            if ch in SENTENCE_END:
                # Absorb any trailing closers (」』）...) into this sentence.
                j = i + 1
                while j < n and run[j] in _CLOSERS:
                    buf.append(run[j])
                    j += 1
                line = "".join(buf).strip()
                buf = []
                i = j
                if _is_good_line(line, ended=True):
                    yield line
                continue
            i += 1
        tail = "".join(buf).strip()
        if _is_good_line(tail, ended=False):
            yield tail


def _is_good_line(line: str, ended: bool) -> bool:
    # Strip a leading orphan closer that can survive (e.g. run started mid-quote).
    line = line.lstrip("".join(_CLOSERS))
    if len(line) < 6:
        return False
    han = sum(1 for c in line if re.match(CJK, c))
    if han < 5 or han / len(line) < 0.6:
        return False
    # Fragments lacking sentence-end punctuation are usually structural noise;
    # keep only the longer, denser ones that look like real prose.
    if not ended:
        return len(line) >= 16 and han / len(line) >= 0.85
    return True


def iter_corpus(limit_docs: int | None):
    """Round-robin stream across all datasets, cleaning and splitting into
    sentence-ish lines, so the output budget is SHARED across sources instead of
    being filled entirely by whichever dataset comes first. The caller honors the
    output-size budget (after dedup) and stops iteration; this just yields lines.
    """
    from datasets import load_dataset

    # Open one streaming iterator per dataset.
    sources = []
    for name, config, split, field in DATASETS:
        print(f"[prepare] streaming {name} (field={field})", file=sys.stderr)
        try:
            ds = load_dataset(name, config, split=split, streaming=True)
        except Exception as e:  # noqa: BLE001
            print(f"[prepare] WARN could not load {name}: {e}", file=sys.stderr)
            continue
        sources.append({"name": name, "field": field, "it": iter(ds),
                        "docs": 0})

    # One document per source per cycle; drop a source when it is exhausted or
    # hits limit_docs. (Wiki articles are longer than news items, so char
    # contribution still skews toward the longer source, but every source is now
    # represented rather than only the first.)
    while sources:
        still_active = []
        for src in sources:
            try:
                row = next(src["it"])
            except StopIteration:
                print(f"[prepare] {src['name']}: exhausted at {src['docs']} docs",
                      file=sys.stderr)
                continue
            raw = row.get(src["field"]) or row.get("text") or ""
            if raw:
                cleaned = clean_text(raw)
                for line in split_sentences(cleaned):
                    yield line
            src["docs"] += 1
            if limit_docs and src["docs"] >= limit_docs:
                print(f"[prepare] {src['name']}: hit limit-docs {limit_docs}",
                      file=sys.stderr)
                continue
            still_active.append(src)
        sources = still_active


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="corpus.txt")
    ap.add_argument("--max-chars", type=lambda s: int(s.replace("_", "")),
                    default=50_000_000,
                    help="stop after this many output characters")
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="optional cap on docs per dataset (for quick tests)")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false",
                    help="keep exact-duplicate lines (default: drop them). "
                         "Repeated lines (wiki boilerplate, copy-pasted "
                         "fragments) bias the LM toward those exact strings.")
    args = ap.parse_args()

    # Exact-line dedup, streaming. We store 64-bit hashes rather than the full
    # lines so the seen-set stays ~64MB even at tens of millions of lines; a
    # hash collision would at worst drop one extra legitimate line (probability
    # is negligible at our scale, and the cost of one dropped line is nil).
    seen: set[int] = set()
    n_lines = 0
    n_chars = 0
    n_dups = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for line in iter_corpus(args.limit_docs):
            if args.dedup:
                h = hash(line)
                if h in seen:
                    n_dups += 1
                    continue
                seen.add(h)
            f.write(line)
            f.write("\n")
            n_lines += 1
            n_chars += len(line) + 1
            if n_lines % 100_000 == 0:
                print(f"[prepare] {n_lines} lines, {n_chars} chars "
                      f"({n_dups} dups dropped)", file=sys.stderr)
            # Budget is enforced here, on deduplicated output, so corpus.txt
            # actually reaches --max-chars of distinct training text.
            if n_chars >= args.max_chars:
                print(f"[prepare] reached max-chars={args.max_chars}",
                      file=sys.stderr)
                break

    print(f"[prepare] DONE: {n_lines} lines, {n_chars} chars "
          f"({n_dups} exact-duplicate lines dropped) -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
