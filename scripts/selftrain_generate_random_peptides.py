#!/usr/bin/env python3
"""Generate random peptide sequences in FASTA format."""

import random
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate random peptide sequences.")
    parser.add_argument("--output", default="random_peptides.fasta", help="Output FASTA path")
    parser.add_argument("--count", type=int, default=10_000_000, help="Number of peptides")
    parser.add_argument("--min_len", type=int, default=3, help="Minimum peptide length")
    parser.add_argument("--max_len", type=int, default=10, help="Maximum peptide length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"

    with open(args.output, "w") as f:
        for i in range(1, args.count + 1):
            length = random.randint(args.min_len, args.max_len)
            peptide = "".join(random.choice(aa) for _ in range(length))
            f.write(f">{i}\n{peptide}\n")

    print(f"Generated {args.count:,} peptides -> {args.output}")


if __name__ == "__main__":
    main()
