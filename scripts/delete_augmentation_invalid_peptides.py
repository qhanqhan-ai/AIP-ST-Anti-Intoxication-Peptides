#!/usr/bin/env python3
"""Filter invalid peptide records from a JSONL file."""

import argparse
import json
import re
from pathlib import Path


VALID_PEPTIDE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def is_valid_peptide(sequence):
    return bool(VALID_PEPTIDE.fullmatch(sequence or ""))


def main():
    parser = argparse.ArgumentParser(description="Filter invalid peptide records from JSONL.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--sequence-key", default="query", help="Sequence field name.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    removed = 0
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sequence = str(record.get(args.sequence_key, "")).strip().upper()
            if not is_valid_peptide(sequence):
                removed += 1
                print(f"[FILTERED] line={line_number} sequence={sequence} target={record.get('target')}")
                continue
            record[args.sequence_key] = sequence
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[SUMMARY] kept={kept} removed={removed} output={output_path}")


if __name__ == "__main__":
    main()
