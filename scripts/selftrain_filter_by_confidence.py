#!/usr/bin/env python3
"""Filter model predictions by confidence threshold and sample per-class subsets.

Uses reservoir sampling for memory-efficient streaming on large CSV/TSV files.
"""

import csv
import json
import random
import argparse


def detect_sep(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    return "\t" if head.count("\t") >= head.count(",") else ","


def process_stream(input_path, output_path, prob_col, target, thresh, sample_size, seed, sep):
    random.seed(seed)
    reservoir = []
    total_read = 0
    kept_total = 0

    with open(input_path, "r", encoding="utf-8", newline="") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter=sep, quoting=csv.QUOTE_NONE)
        for row in reader:
            total_read += 1
            try:
                prob = float(row[prob_col])
            except (ValueError, KeyError):
                continue
            if prob > thresh:
                seq = row.get("seq", "")
                kept_total += 1
                if len(reservoir) < sample_size:
                    reservoir.append(seq)
                else:
                    r = random.randint(0, kept_total - 1)
                    if r < sample_size:
                        reservoir[r] = seq

        for seq in reservoir:
            fout.write(json.dumps({"query": seq, "target": target}) + "\n")

    return total_read, kept_total, len(reservoir)


def main():
    parser = argparse.ArgumentParser(description="Filter predictions by confidence and sample")
    parser.add_argument("--input", required=True, help="Input CSV/TSV")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--prob_col", required=True, help="Probability column (e.g. P[0])")
    parser.add_argument("--target", required=True, help="Target label for filtered sequences")
    parser.add_argument("--thresh", type=float, default=0.90, help="Confidence threshold")
    parser.add_argument("--sample", type=int, default=3000, help="Max samples to keep")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--sep", default=None, help="Column separator (auto-detect if omitted)")
    args = parser.parse_args()

    sep = args.sep if args.sep else detect_sep(args.input)
    total, kept, written = process_stream(
        args.input, args.output, args.prob_col, args.target,
        args.thresh, args.sample, args.seed, sep
    )
    print(f"Done: {total:,} read, {kept:,} passed, {written:,} sampled -> {args.output}")


if __name__ == "__main__":
    main()
