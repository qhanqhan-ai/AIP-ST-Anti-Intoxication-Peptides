# Workflow Summary

## 1. Raw Dataset Construction

The raw dataset used before augmentation is:

`data/raw_preaugmentation_clean01_plus_class3_class5.jsonl`

It was constructed by merging cleaned target 0/1 samples with target 3/5 samples from the augmentation source dataset.

Class distribution:

| target | count |
|---|---:|
| 0 | 1142 |
| 1 | 933 |
| 3 | 103 |
| 5 | 42 |
| total | 2220 |

## 2. Data Augmentation

The main augmentation script included in this package is:

`scripts/augment_to_1508_unique.py`

The augmentation is rule-based for short peptides. It includes:

- conservative amino-acid substitution by physicochemical groups
- alanine scanning
- local sequence shuffle
- sequence reversal

## 3. Molecular Docking Filtering

Augmented target 3 and target 5 peptides were filtered by molecular docking. Peptides with docking binding energy < 5 were retained and merged into the training dataset.

The docking-filtered base training dataset is:

`data/docking_filtered_base_train_3528.jsonl`

Class distribution:

| target | count |
|---|---:|
| 0 | 1142 |
| 1 | 933 |
| 3 | 819 |
| 5 | 634 |
| total | 3528 |

## 4. Feature Extraction

All model features were extracted using ProtBERT. The downstream model inputs are ProtBERT embeddings of peptide sequences.

## 5. Baseline and Deep Learning Model Comparison

The compared models were:

- LogReg
- SVM
- RF
- MLP
- CNN
- Transformer

Main evaluation criteria:

- Macro-F1
- Class 3 F1
- Class 3 precision

All deep learning models (MLP, CNN, Transformer) were trained with identical training strategy:
- OneCycleLR learning rate scheduler
- EMA (Exponential Moving Average, decay=0.999)
- Early stopping (patience=30, monitor=val_loss)

The final comparison table is in:

`results/model_metrics_summary.csv`

## 6. Transformer Selection

The Transformer model was selected because it achieved the best balance between overall Macro-F1, Class 3 F1, and Class 3 precision.

The selected Transformer (best among 5-seed ensemble) reached:

- Macro-F1: 0.7424 (mean)
- Class 3 precision: 0.6892 (mean)

This model was then used for self-training.

## 7. Self-training

A total of 10 million random peptide sequences were generated and scored by the selected Transformer model. High-confidence predictions were filtered:

- confidence >= 0.90
- 3000 samples per class
- deduplicated before merging

The resulting self-training dataset is:

`data/self_training_plus_3000_per_class_deduplicated.jsonl`

## 8. Final Prediction

The final self-trained Transformer model was used to predict:

`data/final_inference_input.fasta`

The final prediction file is:

`results/final_predictions.csv`

Example final prediction:

```csv
id,seq,pred_id,pred_label,pred_conf,P[0],P[1],P[3],P[5]
2011,GLKLL,3,5,0.98291,0.0131,0.001793,0.002092,0.98291
```
