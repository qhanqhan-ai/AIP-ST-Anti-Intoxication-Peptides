# Local Path Reference

This file records the workflow components that were used locally. Reviewer-facing files in this package use portable English names and package-relative paths.

## Raw and Augmented Data

- Raw pre-augmentation dataset: `data/raw_preaugmentation_clean01_plus_class3_class5.jsonl`
- Data augmentation scripts: `scripts/`
- Molecular docking workflow: external local docking workspace
- Docking-filtered base training dataset: `data/docking_filtered_base_train_3528.jsonl`

## Feature Extraction

- ProtBERT feature extraction project: `/home/csl24/prot_bert`

## Self-training

- The 10-million random peptide FASTA was used as an external large input and is not bundled.
- Deduplicated self-training dataset with 3000 high-confidence samples per class: `data/self_training_plus_3000_per_class_deduplicated.jsonl`

## Final Inference

- Final input FASTA: `data/final_inference_input.fasta`
- Final prediction result: `results/final_predictions.csv`
