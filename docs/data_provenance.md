# Data Provenance

## Included Data Files

| File | Description |
|---|---|
| `data/raw_preaugmentation_clean01_plus_class3_class5.jsonl` | Raw pre-augmentation dataset used in this workflow. |
| `data/docking_filtered_base_train_3528.jsonl` | Base training set after augmentation and molecular docking filtering. |
| `data/self_training_plus_3000_per_class_deduplicated.jsonl` | Self-training dataset after adding 3000 high-confidence samples per class. |
| `data/final_inference_input.fasta` | Final FASTA input for inference. |
| `data/dataset_statistics.tsv` | Machine-readable class distribution summary. |

## Class Distributions

### Raw Pre-augmentation Dataset

`data/raw_preaugmentation_clean01_plus_class3_class5.jsonl`

| target | count |
|---|---:|
| 0 | 1142 |
| 1 | 933 |
| 3 | 103 |
| 5 | 42 |
| total | 2220 |

### Docking-filtered Base Training Set

`data/docking_filtered_base_train_3528.jsonl`

| target | count |
|---|---:|
| 0 | 1142 |
| 1 | 933 |
| 3 | 819 |
| 5 | 634 |
| total | 3528 |

### Self-training Dataset

`data/self_training_plus_3000_per_class_deduplicated.jsonl`

| target | count |
|---|---:|
| 0 | 4142 |
| 1 | 3933 |
| 3 | 3819 |
| 5 | 3634 |
| total | 15528 |

## Notes

- `data/docking_filtered_base_train_3528.jsonl` is the docking-filtered base dataset used for feature extraction and model training.
- The large 10-million random peptide FASTA file is intentionally not included in this GitHub package due to size.
