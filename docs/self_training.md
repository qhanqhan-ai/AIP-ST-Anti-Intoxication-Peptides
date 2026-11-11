# Self-training Details

## Overview

After selecting the Transformer model, self-training was applied to expand the training dataset using high-confidence predictions on randomly generated peptide sequences.

## Pipeline

### Step 1: Random Peptide Generation

Script: `scripts/selftrain_generate_random_peptides.py`

Generated 10 million random peptide sequences with the following parameters:

- Amino acid pool: 20 standard amino acids
- Peptide length: 3–10 (random)
- Random seed: 42

Output: `random_peptides.fasta` (10,000,000 sequences)

### Step 2: ProtBERT Feature Extraction

All 10 million random peptide sequences were embedded using ProtBERT, producing ProtBERT embeddings for each sequence.

### Step 3: Inference with Selected Transformer

Script: `scripts/selftrain_infer.py`

The selected Transformer model (seed5) was used to predict class probabilities for all 10 million embedded sequences.

### Step 4: Confidence Filtering

Script: `scripts/selftrain_filter_by_confidence.py`

For each class (0, 1, 3, 5), sequences with prediction confidence >= 0.90 were retained. Up to 3000 sequences per class were randomly sampled from the high-confidence pool.

- Confidence threshold: 0.90
- Samples per class: 3000
- Sampling method: Reservoir sampling (ensures equal probability)
- Random seed: 1

### Step 5: Deduplication and Merging

Script: `scripts/selftrain_deduplicate.py`

The filtered self-labeled sequences were deduplicated by peptide sequence and merged with the original training data.

Final self-training dataset: `data/self_training_plus_3000_per_class_deduplicated.jsonl` (15,528 samples)

### Step 6: Retrain Transformer

The Transformer was retrained on the expanded dataset (original 3528 + self-labeled 12000 = 15528 samples).

Self-trained Transformer results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.9487 | 0.9614 | 0.9550 | 828 |
| 1 | 0.9478 | 0.9682 | 0.9579 | 787 |
| 3 | 0.9440 | 0.9267 | 0.9353 | 764 |
| 5 | 0.9790 | 0.9601 | 0.9694 | 727 |
| **Macro avg** | **0.9549** | **0.9541** | **0.9544** | 3106 |

Overall accuracy: 0.9543
Macro ROC-AUC (OVR): 0.9963

## Scripts

| Script | Description |
|---|---|
| `selftrain_generate_random_peptides.py` | Generate random peptide sequences |
| `selftrain_infer.py` | Run inference on random peptides |
| `selftrain_filter_by_confidence.py` | Filter by confidence threshold (per class) |
| `selftrain_filter_merge_dedup.py` | Combined filter, merge, and deduplicate |
| `selftrain_deduplicate.py` | Deduplicate JSONL by sequence |
