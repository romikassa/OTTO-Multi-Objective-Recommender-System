# OTTO – Multi-Objective Recommender System

Kaggle competition: predict clicks, carts and orders from e-commerce sessions.

## Results

| Version | Method | Private score |
|---|---|---|
| 1 | Baseline: top 20 popular items | 0.00550 |
| 2 | Co-visitation | 0.10191 |
| 3 | + last session items kept first | 0.45216 |
| 4 | + word2vec neighbors (dim 64, SEEN_N=10) | 0.50532 |

## Files
- `otto_recommender.ipynb`: the Kaggle notebook
