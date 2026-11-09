## Test Set Results

Task: predict whether a conjunction event peaks at high risk (event max Pc >= 1e-3). Test set: 111 events, 18 high-risk (16.2%).

| Model | AUC-ROC | Avg Precision | Accuracy | Precision | Recall | F1 (thr=0.5) |
|---|---|---|---|---|---|---|
| RetNet (ours) | 0.783 | 0.385 | 0.829 | 0.478 | 0.611 | 0.537 |
| Transformer | 0.873 | 0.601 | 0.802 | 0.433 | 0.722 | 0.542 |

*Metrics at threshold 0.5. Threshold-tuned results are in `evaluation_summary.json`.*
