# OTTO – Multi-Objective Recommender System

Solution for the Kaggle competition [OTTO – Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system).
The goal is to predict, for each e-commerce session, which items the user will **click**, **add to cart** and **order** next (20 predictions per session and per action type).

- Kaggle notebook: https://www.kaggle.com/code/roumaisaemeliani/notebook-of-otto-s-mors-competition
- Metric: weighted recall@20 (clicks 10 %, carts 30 %, orders 60 %)
- Note: the competition is over, so all submissions were made as late submissions ("Complete (after deadline)").

## Results

Each row is one submission. Scores are the Kaggle **private** scores.

| # | Method | Private score |
|---|--------|---------------|
| 1 | Baseline: the 20 most popular items, same list for every session | 0.00550 |
| 2 | Co-visitation matrix (last 20 events per session, recency-weighted), filled with popular items | 0.10191 |
| 3 | Co-visitation + the last 5 items of the session kept first (`SEEN_N=5`) | 0.45216 |
| 4 | Co-visitation + last 10 session items (`SEEN_N=10`) + word2vec neighbors to fill remaining slots (dim 64) | 0.50532 |
| 5 | Final notebook: 1-day time window, type weights, buy-only matrix, separate lists for clicks vs carts/orders, word2vec fill | 0.53285 |

Keeping the items already seen in the session at the top of the list was the biggest single gain (0.10 → 0.45), which shows that users often come back to items they already looked at.

## Method

1. **Baseline** – count the most frequent items in a sample of the training sessions and recommend the same top 20 to everyone.
2. **Co-visitation matrix** – two items that appear in the same sessions are considered related. For each item we keep its 20 most co-visited items. A session's recent items count more than old ones (weight `1 / (1 + rank)`).
3. **Items from the current session** – the last `SEEN_N` items of the session are placed at the top of the list, then the co-visitation candidates.
4. **word2vec fill** – a word2vec model is trained on the sessions (each session is a "sentence", each item a "word"). When a session has fewer than 20 candidates, the list is completed with the nearest items in the embedding space, then with popular items if needed.
5. **Final version improvements**
   - co-visitation restricted to events less than **1 day** apart;
   - pairs weighted by event type (click = 1, cart = 6, order = 3);
   - a second matrix built **only from carts and orders** (14-day window), used for the `carts` and `orders` predictions;
   - a **different list for `clicks`** and for **`carts` / `orders`**: for carts and orders, items already carted or ordered in the session come first.

## Parameters of the final notebook

| Parameter | Value |
|-----------|-------|
| Training sessions used (`MAX_SESSIONS`) | 1,000,000 (39.9 M events) |
| Last events per session (`LAST_N`) | 20 |
| Neighbors kept per item (`TOP_PER_ITEM`) | 20 |
| Items of the session put first (`SEEN_N`) | 10 |
| Time window / buy window | 1 day / 14 days |
| Type weights (click / cart / order) | 1 / 6 / 3 |
| Buy-matrix weight (`BUY_W`) | 2.0 |
| word2vec | skip-gram, dim 64, window 5, min_count 5, 3 epochs |

The weights (type weights and `BUY_W`) are starting values, not tuned values.

## Runtime (from the notebook run, CPU)

| Step | Time |
|------|------|
| Load 1 M training sessions | ~1 min |
| Co-visitation matrices (all + buy-only) | ~2.5 min |
| word2vec training + neighbors for ~813 k items | ~1 h 30 |
| Recommendations and `submission.csv` | ~3 min |

word2vec is by far the slowest step.

## How to run

1. Open the notebook on Kaggle and attach the competition data (**Add Input** → *OTTO – Multi-Objective Recommender System*). The files are read from `/kaggle/input/competitions/otto-recommender-system/`.
2. Run the cells in order: parameters and loading → co-visitation → word2vec → recommendations.
3. Submit the generated `submission.csv` from the **Output** tab.

## Repository content

- `notebook-of-otto-s-mors-competition.ipynb` – the Kaggle notebook.
- The OTTO data and the generated `submission.csv` (about 760 MB) are **not** included: they are too large for GitHub and the data belongs to the competition.

## Limitations and next steps

- Only a sample (1 M) of the training sessions is used, and no local validation set was built: the only evaluation is the Kaggle score.
- The strongest solutions of the competition use a two-stage pipeline: candidate generation from several matrices and embeddings, then a gradient-boosting ranker (LightGBM / XGBoost) trained on a held-out week. That ranker is not implemented here.
- `carts` and `orders` share the same list; separating them is a possible improvement.
