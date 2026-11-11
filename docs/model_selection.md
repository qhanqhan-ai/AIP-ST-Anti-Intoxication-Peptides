# Model Selection Summary

## Model Comparison (5 seeds: 2, 3, 5, 11, 13)

All models were trained with the same experimental setup:
- **Seeds**: 2, 3, 5, 11, 13
- **Data**: protbert.jsonl (3528 samples, ProtBERT embeddings)
- **Test ratio**: 0.2 (stratified)
- **Deep learning params**: batch_size=64, lr=2e-4, weight_decay=1e-2, dropout=0.3, epochs=200
- **Deep learning training strategy**: OneCycleLR scheduler, EMA (decay=0.999), Early stopping (patience=30, monitor=val_loss)
- **Classic model params**: LogReg (C=1.0, max_iter=2000), SVM (kernel=rbf, C=1.0), RF (n_estimators=200, max_depth=20)

### Overall Metrics

| Model | ACC | PRE | REC | SPE | F1 |
|---|---:|---:|---:|---:|---:|
| LogReg | 0.6595 ± 0.0183 | 0.6415 ± 0.0168 | 0.6363 ± 0.0162 | 0.8856 ± 0.0065 | 0.6377 ± 0.0163 |
| SVM | 0.6190 ± 0.0189 | 0.6037 ± 0.0202 | 0.5862 ± 0.0167 | 0.8703 ± 0.0064 | 0.5876 ± 0.0165 |
| RF | 0.6861 ± 0.0211 | 0.6819 ± 0.0214 | 0.6628 ± 0.0188 | 0.8931 ± 0.0073 | 0.6681 ± 0.0190 |
| MLP | 0.7040 ± 0.0099 | 0.6964 ± 0.0109 | 0.6908 ± 0.0102 | 0.9004 ± 0.0035 | 0.6928 ± 0.0103 |
| CNN | 0.7326 ± 0.0126 | 0.7246 ± 0.0102 | 0.7211 ± 0.0118 | 0.9104 ± 0.0044 | 0.7224 ± 0.0111 |
| Transformer | 0.7550 ± 0.0115 | 0.7590 ± 0.0150 | 0.7386 ± 0.0172 | 0.9169 ± 0.0046 | 0.7428 ± 0.0148 |

### Per-Class F1

| Model | Class 0 | Class 1 | Class 3 | Class 5 |
|---|---:|---:|---:|---:|
| LogReg | 0.7784 ± 0.0323 | 0.6732 ± 0.0193 | 0.5624 ± 0.0265 | 0.5366 ± 0.0268 |
| SVM | 0.7417 ± 0.0252 | 0.6260 ± 0.0184 | 0.5354 ± 0.0339 | 0.4472 ± 0.0335 |
| RF | 0.7803 ± 0.0321 | 0.6700 ± 0.0255 | 0.6261 ± 0.0296 | 0.5958 ± 0.0338 |
| MLP | 0.7849 ± 0.0217 | 0.7148 ± 0.0225 | 0.6182 ± 0.0230 | 0.6535 ± 0.0381 |
| CNN | 0.8158 ± 0.0188 | 0.7378 ± 0.0218 | 0.6472 ± 0.0173 | 0.6888 ± 0.0183 |
| Transformer | 0.8202 ± 0.0252 | 0.7774 ± 0.0136 | 0.6645 ± 0.0216 | 0.7093 ± 0.0439 |

### Per-Class Precision

| Model | Class 0 | Class 1 | Class 3 | Class 5 |
|---|---:|---:|---:|---:|
| LogReg | 0.7459 ± 0.0362 | 0.6935 ± 0.0232 | 0.5607 ± 0.0253 | 0.5660 ± 0.0384 |
| SVM | 0.6813 ± 0.0213 | 0.6448 ± 0.0239 | 0.5187 ± 0.0219 | 0.5698 ± 0.0418 |
| RF | 0.7350 ± 0.0313 | 0.6625 ± 0.0333 | 0.6310 ± 0.0326 | 0.6989 ± 0.0485 |
| MLP | 0.7747 ± 0.0229 | 0.7288 ± 0.0312 | 0.6004 ± 0.0134 | 0.6815 ± 0.0522 |
| CNN | 0.8152 ± 0.0138 | 0.7499 ± 0.0285 | 0.6254 ± 0.0226 | 0.7081 ± 0.0237 |
| Transformer | 0.8080 ± 0.0447 | 0.7405 ± 0.0115 | 0.6795 ± 0.0561 | 0.8082 ± 0.0772 |

## Selected Model

The Transformer model was selected for downstream self-training because it achieved the best balance between overall Macro-F1, Class 3 F1, Class 3 precision, and Class 3 AP.

## Training Configuration

All six models share the same experimental conditions:
- Same dataset and data split (seed-controlled, stratified 80/20)
- Same 5 random seeds for reproducibility
- Deep learning models (MLP, CNN, Transformer) use identical training hyperparameters and strategy (OneCycleLR, EMA, Early Stopping)
- Classic models (LogReg, SVM, RF) use their respective sklearn defaults
