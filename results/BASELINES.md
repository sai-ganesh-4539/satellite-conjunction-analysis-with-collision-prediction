## Baseline Comparison (Test Set)

| Model | AUC-ROC | Avg Precision | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Majority class | 0.500 | 0.254 | 0.746 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.860 | 0.694 | 0.786 | 0.564 | 0.688 | 0.620 |
| Random forest | 0.893 | 0.762 | 0.857 | 0.750 | 0.656 | 0.700 |
| Gradient boosting | 0.935 | 0.882 | 0.929 | 0.926 | 0.781 | 0.848 |

*All models use the identical train/test split (seed 42). Classical models receive the same 12x15 CDM sequences flattened into 180 features. Deep-model numbers are from `train.py` (best checkpoints).*
