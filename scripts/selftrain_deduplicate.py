#!/usr/bin/env python3
"""Deduplicate JSONL records by peptide sequence.

Removes duplicate entries based on the 'query' field, keeping only the
first occurrence of each unique sequence.
"""

import json
import gzip
import argparse
from collections import Counter


def smart_open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate JSONL by peptide sequence")
    parser.add_argument("--input", required=True, help="Input JSONL (may be .gz)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--field", default="query", help="Deduplication key field")
    args = parser.parse_args()

    seen = set()
    total = unique = dup = bad = 0
    stats = Counter()

    with smart_open(args.input) as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                key = obj.get(args.field, "")
                if not key or key in seen:
                    dup += 1
                    continue
                seen.add(key)
                fout.write(s + "\n")
                unique += 1
                if "target" in obj:
                    stats[obj["target"]] += 1
            except (json.JSONDecodeError, Exception):
                bad += 1

    print(f"Deduplication: {total:,} total, {unique:,} unique, {dup:,} dup, {bad:,} bad")
    for label in sorted(stats.keys()):
        print(f"  class {label}: {stats[label]:,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
