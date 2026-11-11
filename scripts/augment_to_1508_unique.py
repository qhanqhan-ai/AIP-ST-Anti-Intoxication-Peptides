#!/usr/bin/env python3
"""Rule-based unique peptide augmentation for short peptides.

The script keeps locked classes unchanged and augments target classes up to a
requested unique per-class count. Augmentation preserves sequence length.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


AA20 = set("ACDEFGHIKLMNPQRSTVWY")
SUBSTITUTION_GROUPS = [
    list("VILMAF"),
    list("STYWNQ"),
    list("KRH"),
    list("DE"),
    list("GP"),
    list("C"),
]


def group_for(residue):
    for group in SUBSTITUTION_GROUPS:
        if residue in group:
            return group
    return [residue]


def conservative_substitution(seq):
    positions = [idx for idx, residue in enumerate(seq) if len(group_for(residue)) > 1]
    if not positions:
        return seq
    idx = random.choice(positions)
    residue = seq[idx]
    choices = [candidate for candidate in group_for(residue) if candidate != residue]
    return seq[:idx] + random.choice(choices) + seq[idx + 1 :]


def alanine_scan(seq):
    positions = [idx for idx, residue in enumerate(seq) if residue != "A"]
    if not positions:
        return seq
    idx = random.choice(positions)
    return seq[:idx] + "A" + seq[idx + 1 :]


def local_shuffle(seq):
    length = len(seq)
    if length < 5:
        return seq
    window = max(2, min(max(2, int(0.3 * length)), length - 2))
    start = random.randint(1, length - window - 1)
    segment = list(seq[start : start + window])
    shuffled = segment[:]
    for _ in range(8):
        random.shuffle(shuffled)
        if shuffled != segment:
            break
    return seq[:start] + "".join(shuffled) + seq[start + window :]


def reverse_sequence(seq):
    return seq[::-1]


def augment_once(seq):
    methods = [
        ("substitution", conservative_substitution, 0.46),
        ("alanine_scan", alanine_scan, 0.28),
        ("local_shuffle", local_shuffle, 0.16 if len(seq) >= 5 else 0.0),
        ("reverse", reverse_sequence, 0.10),
    ]
    active = [(name, func, weight) for name, func, weight in methods if weight > 0]
    names, funcs, weights = zip(*active)
    for _ in range(6):
        idx = random.choices(range(len(active)), weights=weights, k=1)[0]
        candidate = funcs[idx](seq)
        if candidate != seq:
            return candidate, names[idx]
    return seq, "unchanged"


def read_jsonl(path, min_len, max_len):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sequence = str(record.get("query", record.get("sequence", ""))).strip().upper()
            target = str(record.get("target", record.get("label", ""))).strip()
            if min_len <= len(sequence) <= max_len and set(sequence) <= AA20 and target:
                records.append({"query": sequence, "target": target})
            else:
                print(f"[SKIP] line={line_number} sequence={sequence} target={target}")
    return records


def add_unique(output_records, seen, sequence, target, source, original):
    key = (target, sequence)
    if key in seen:
        return False
    seen.add(key)
    output_records.append(
        {
            "query": sequence,
            "target": target,
            "augment_source": source,
            "orig_query": original,
        }
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Create unique rule-based peptide augmentations.")
    parser.add_argument("--in-jsonl", required=True, help="Input JSONL file with query and target fields.")
    parser.add_argument("--out-jsonl", required=True, help="Output augmented JSONL file.")
    parser.add_argument("--targets", nargs="+", default=["5", "3"], help="Target classes to augment.")
    parser.add_argument("--locked", nargs="+", default=["1", "0"], help="Classes kept without augmentation.")
    parser.add_argument("--per-class", type=int, default=1508, help="Maximum unique records per augmented class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iters", type=int, default=300000, help="Maximum attempts per class.")
    parser.add_argument("--patience", type=int, default=800, help="Stop after this many failed unique additions.")
    parser.add_argument("--min-len", type=int, default=3, help="Minimum peptide length.")
    parser.add_argument("--max-len", type=int, default=10, help="Maximum peptide length.")
    args = parser.parse_args()

    random.seed(args.seed)
    targets = set(args.targets)
    locked = set(args.locked)
    records = read_jsonl(args.in_jsonl, args.min_len, args.max_len)

    by_target = defaultdict(list)
    for record in records:
        by_target[record["target"]].append(record["query"])

    output_records = []
    seen = set()
    method_count = Counter()
    for target, sequences in sorted(by_target.items()):
        for sequence in sorted(set(sequences)):
            if add_unique(output_records, seen, sequence, target, "original", sequence):
                method_count["original"] += 1

    for target in sorted(targets):
        base_sequences = list(dict.fromkeys(by_target.get(target, [])))
        if not base_sequences:
            print(f"[WARN] class={target} has no source records")
            continue
        attempts = 0
        failures = 0
        current_count = sum(1 for item in output_records if item["target"] == target)
        while current_count < args.per_class and attempts < args.max_iters and failures < args.patience:
            attempts += 1
            source = random.choice(base_sequences)
            augmented, method = augment_once(source)
            if len(augmented) != len(source) or set(augmented) - AA20:
                failures += 1
                continue
            if add_unique(output_records, seen, augmented, target, method, source):
                method_count[method] += 1
                current_count += 1
                failures = 0
            else:
                failures += 1
        print(f"[INFO] class={target} unique_count={current_count} attempts={attempts}")

    for target in sorted(locked):
        if target in by_target:
            print(f"[LOCKED] class={target} kept without augmentation")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    class_count = Counter(record["target"] for record in output_records)
    print("[SUMMARY] class_counts=", dict(sorted(class_count.items())))
    print("[SUMMARY] method_counts=", method_count.most_common())
    print(f"[DONE] output={out_path}")


if __name__ == "__main__":
    main()
