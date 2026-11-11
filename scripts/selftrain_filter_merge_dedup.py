#!/usr/bin/env python3
"""Filter predictions by confidence, sample per class, merge with existing data, and deduplicate.

Reads a CSV/TSV of model predictions, filters per-class by probability threshold,
samples up to N sequences per class via reservoir sampling, merges with an existing
JSONL training set, and deduplicates by sequence.
"""

import csv
import json
import gzip
import random
import argparse
from collections import Counter


LABEL_COLS = [("0", "P[0]"), ("1", "P[1]"), ("3", "P[3]"), ("5", "P[5]")]


def smart_open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def detect_sep(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    return "\t" if head.count("\t") >= head.count(",") else ","


def copy_existing(existing_path, fout):
    existing_set = set()
    copied = 0
    with smart_open(existing_path) as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                seq = obj.get("query", "")
                if seq:
                    existing_set.add(seq)
                fout.write(s + "\n")
                copied += 1
            except json.JSONDecodeError:
                continue
    return existing_set, copied


def process_stream(input_path, fout, existing_set, thresh, per_label, seed, sep):
    random.seed(seed)
    reservoir = {lab: [] for lab, _ in LABEL_COLS}
    counted = {lab: 0 for lab, _ in LABEL_COLS}
    already_picked = set()

    with open(input_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter=sep, quoting=csv.QUOTE_NONE)
        for row in reader:
            for lab, prob_col in LABEL_COLS:
                try:
                    prob = float(row[prob_col])
                except (ValueError, KeyError):
                    continue
                seq = row.get("seq", "")
                if prob <= thresh or seq in existing_set or seq in already_picked:
                    continue
                counted[lab] += 1
                if len(reservoir[lab]) < per_label:
                    reservoir[lab].append(seq)
                    already_picked.add(seq)
                else:
                    r = random.randint(0, counted[lab] - 1)
                    if r < per_label:
                        already_picked.discard(reservoir[lab][r])
                        reservoir[lab][r] = seq
                        already_picked.add(seq)

    for lab, _ in LABEL_COLS:
        for seq in reservoir[lab]:
            fout.write(json.dumps({"query": seq, "target": lab}) + "\n")

    eligible = {lab: counted[lab] for lab, _ in LABEL_COLS}
    written = {lab: len(reservoir[lab]) for lab, _ in LABEL_COLS}
    return eligible, written


def main():
    parser = argparse.ArgumentParser(description="Filter, sample, merge and deduplicate")
    parser.add_argument("--input", required=True, help="Prediction CSV/TSV")
    parser.add_argument("--existing", required=True, help="Existing training JSONL")
    parser.add_argument("--output", required=True, help="Output merged JSONL")
    parser.add_argument("--thresh", type=float, default=0.90, help="Confidence threshold")
    parser.add_argument("--per_label", type=int, default=3000, help="Samples per class")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--sep", default=None, help="Column separator (auto-detect)")
    args = parser.parse_args()

    sep = args.sep if args.sep else detect_sep(args.input)

    with open(args.output, "w", encoding="utf-8") as fout:
        existing_set, copied = copy_existing(args.existing, fout)
        print(f"Copied {copied:,} existing records; dedup set: {len(existing_set):,}")
        eligible, written = process_stream(
            args.input, fout, existing_set,
            args.thresh, args.per_label, args.seed, sep
        )

    print(f"Per-class eligible (after dedup): {dict(eligible)}")
    print(f"Per-class sampled: {dict(written)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
