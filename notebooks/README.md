# Notebooks

- `01_eda.ipynb` (W1): review volume over time, rating distribution, price vs reception, language mix.
  Start with `uv run --extra notebooks jupyter lab`.
- `02_sentiment.ipynb` (W3): TF-IDF + logistic regression vs the majority class on held-out
  games, ROC, strongest words, aspect distribution. Writes `models/sentiment_tfidf_lr.*`.
- `03_reception.ipynb` (W3): LightGBM reception buckets with a time split, bucket drift by
  year, calibration, SHAP importance. Writes `models/reception_lgbm.*`.
