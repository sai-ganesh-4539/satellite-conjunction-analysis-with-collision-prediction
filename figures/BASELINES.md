## Baseline Comparison (Test Set)

| Model | AUC-ROC | Avg Precision | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Majority class | 0.500 | 0.162 | 0.838 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.795 | 0.582 | 0.811 | 0.440 | 0.611 | 0.512 |
| Random forest | 0.852 | 0.608 | 0.865 | 0.588 | 0.556 | 0.571 |
| Gradient boosting | 0.866 | 0.697 | 0.901 | 0.706 | 0.667 | 0.686 |
| RetNet (ours) | 0.783 | 0.385 | 0.829 | 0.478 | 0.611 | 0.537 |
| Transformer | 0.873 | 0.601 | 0.802 | 0.433 | 0.722 | 0.542 |

*All models use the identical train/test split (seed 42). Classical models receive the same 12x15 CDM sequences flattened into 180 features. Deep-model numbers are from `train.py` (best checkpoints).*
