"""review_sentiment: sentiment and aspect distribution of one game's reviews in a time window.

Works on the English text sample (reviews.parquet), reweighted to the population: the sample
is stratified by year and rating, so each sampled review gets weight
N(game, year, rating) / n(game, year, rating) from review_volume. Shares are weighted means;
tests use the Kish effective sample size.

Per aspect (keyword lexicon), the sentences that mention it are scored by the TF-IDF + LR
classifier, so a positive review can still carry a performance complaint:
  mention_share              share of reviews mentioning the aspect
  negative_share_of_mentions share of those mentions the classifier scores negative
  complaint_share            share of all reviews with a negative mention of the aspect"""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, model_validator

from steam_game_analyst.ml.aspects import ASPECTS, aspect_sentences
from steam_game_analyst.ml.sentiment_cache import aspect_inputs
from steam_game_analyst.tools import stats
from steam_game_analyst.tools.catalog import CatalogDB

LANGUAGE = "english"
MIN_SIDE_DAYS = 28  # a before/after comparison needs about a month on each side
MIN_REVIEWS = 30  # below this the window is flagged as a small sample
EXAMPLES_PER_ASPECT = 2
EXAMPLE_CHARS = 160


class Classifier(Protocol):
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


def _shift(d: date, months: int) -> date:
    idx = d.year * 12 + d.month - 1 + months
    y, m = divmod(idx, 12)
    m += 1
    days = [
        31,
        29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ][m - 1]
    return date(y, m, min(d.day, days))


class SentimentArgs(BaseModel):
    appid: int = Field(description="Steam appid of the game")
    date_from: date | None = Field(None, description="Window start (inclusive)")
    date_to: date | None = Field(None, description="Window end (inclusive)")
    split_date: date | None = Field(
        None,
        description="Optional patch/event date: compare aspects before vs after it. Leave "
        "date_from/date_to out to use window_months on each side.",
    )
    window_months: int = Field(6, ge=1, le=36, description="Months on each side of split_date")

    window_adjusted: bool = Field(False, exclude=True)

    @model_validator(mode="after")
    def _ordered(self) -> SentimentArgs:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from is after date_to")
        if self.split_date:
            before_ok = self.date_from and (self.split_date - self.date_from).days >= MIN_SIDE_DAYS
            after_ok = self.date_to and (self.date_to - self.split_date).days >= MIN_SIDE_DAYS
            explicit = self.date_from is not None or self.date_to is not None
            if not (before_ok and after_ok):
                # a comparison needs both sides: fall back to window_months around the date
                self.date_from = _shift(self.split_date, -self.window_months)
                self.date_to = _shift(self.split_date, self.window_months)
                self.window_adjusted = explicit
        return self


def stratum_weights(sample: pl.DataFrame, population: pl.DataFrame) -> pl.DataFrame:
    """sample (appid, year, voted_up, ...) + population (appid, year, n_reviews, n_positive)
    -> sample with a `weight` column: population stratum size / sampled stratum size."""
    pop = pl.concat(
        [
            population.select(
                "appid", "year", pl.lit(True).alias("voted_up"), pl.col("n_positive").alias("N")
            ),
            population.select(
                "appid",
                "year",
                pl.lit(False).alias("voted_up"),
                (pl.col("n_reviews") - pl.col("n_positive")).alias("N"),
            ),
        ]
    ).with_columns(pl.col("year").cast(pl.Int32), pl.col("N").cast(pl.Float64))
    counts = sample.group_by("appid", "year", "voted_up").agg(pl.len().alias("n"))
    return (
        sample.with_columns(pl.col("year").cast(pl.Int32))
        .join(counts.with_columns(pl.col("year").cast(pl.Int32)), on=["appid", "year", "voted_up"])
        .join(pop, on=["appid", "year", "voted_up"], how="left")
        # a stratum missing from the population table keeps weight 1 and is flagged
        .with_columns(
            (pl.col("N").fill_null(pl.col("n")) / pl.col("n")).alias("weight"),
            pl.col("N").is_null().alias("unweighted"),
        )
        .drop("n", "N")
    )


def _wmean(values: np.ndarray, w: np.ndarray) -> float | None:
    return float((values * w).sum() / w.sum()) if w.sum() > 0 else None


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


class ReviewSentiment:
    def __init__(
        self,
        db: CatalogDB,
        reviews_path: Path,
        model: Classifier,
        model_name: str = "TF-IDF+LR classifier",
    ) -> None:
        self.db = db
        self.model = model
        self.model_name = model_name
        self.reviews = pl.scan_parquet(reviews_path).select(
            "review_id", "appid", "language", "review_text", "voted_up", "created_at"
        )

    def _population(self, appid: int) -> pl.DataFrame:
        rows = self.db.fetch(
            "SELECT year(month) AS year, sum(n_reviews), sum(n_positive) FROM review_monthly "
            "WHERE appid = ? GROUP BY 1",
            [appid],
        )
        return pl.DataFrame(
            {
                "appid": [appid] * len(rows),
                "year": [r[0] for r in rows],
                "n_reviews": [int(r[1]) for r in rows],
                "n_positive": [int(r[2]) for r in rows],
            },
            schema={
                "appid": pl.Int64,
                "year": pl.Int32,
                "n_reviews": pl.Int64,
                "n_positive": pl.Int64,
            },
        )

    def _scored(self, df: pl.DataFrame) -> dict[str, Any]:
        """Per review: aspect -> (mentioned, negative mention); plus example sentences."""
        texts = df["review_text"].to_list()
        per_review = [aspect_sentences(t) for t in texts]
        # the same strings the precomputed score table holds (ml/sentiment_cache.py)
        jobs = [(i, a, x) for i, t in enumerate(texts) for a, x in aspect_inputs(t).items()]
        probs = self.model.predict_proba([j[2] for j in jobs]) if jobs else np.zeros(0)
        n = len(texts)
        mention = {a: np.zeros(n) for a in ASPECTS}
        negative = {a: np.zeros(n) for a in ASPECTS}
        for (i, a, _), p in zip(jobs, probs, strict=True):
            mention[a][i] = 1.0
            negative[a][i] = float(p < 0.5)
        return {"mention": mention, "negative": negative, "sentences": per_review}

    def _aspects(self, df: pl.DataFrame, scored: dict[str, Any], w: np.ndarray) -> dict:
        out: dict[str, Any] = {}
        ids = df["review_id"].to_list()
        dates = df["created_at"].to_list()
        for a in ASPECTS:
            m, neg = scored["mention"][a], scored["negative"][a]
            mentioned_w = w * m
            examples = [
                {
                    "review_id": ids[i],
                    "date": dates[i].date().isoformat() if dates[i] else None,
                    "text": " ".join(scored["sentences"][i][a])[:EXAMPLE_CHARS],
                }
                for i in np.argsort(-(w * neg))[:EXAMPLES_PER_ASPECT]
                if neg[i]
            ]
            out[a] = {
                "n_mentions": int(m.sum()),
                "mention_share": _r(_wmean(m, w)),
                "negative_share_of_mentions": _r(_wmean(neg, mentioned_w)) if m.any() else None,
                "complaint_share": _r(_wmean(neg, w)),
                "examples": examples,
            }
        return out

    def run(self, args: SentimentArgs) -> dict[str, Any]:
        names = self.db.fetch("SELECT name FROM games WHERE appid = ?", [args.appid])
        game = {"appid": args.appid, "name": names[0][0] if names else None}
        # weights come from the game's whole sample (every language, every date): stratum
        # counts taken after the window or language filter would inflate them
        full = (
            self.reviews.filter(
                (pl.col("appid") == args.appid) & pl.col("created_at").is_not_null()
            )
            .collect()
            .with_columns(pl.col("created_at").dt.year().alias("year"))
        )
        weighted = stratum_weights(full, self._population(args.appid)) if full.height else full
        q = weighted.filter(pl.col("language") == LANGUAGE)
        if args.date_from:
            q = q.filter(pl.col("created_at") >= datetime.combine(args.date_from, time.min))
        if args.date_to:
            q = q.filter(pl.col("created_at") <= datetime.combine(args.date_to, time.max))
        df = q
        if df.is_empty():
            return {
                "game": game,
                "error": f"no {LANGUAGE} reviews in the sample for appid {args.appid} "
                f"between {args.date_from} and {args.date_to}",
            }
        df = df.sort("created_at")
        w = df["weight"].to_numpy()
        up = df["voted_up"].to_numpy().astype(float)
        scored = self._scored(df)
        prob = self.model.predict_proba(df["review_text"].to_list())
        out: dict[str, Any] = {
            "game": game,
            "window": {
                "from": df["created_at"].min().date().isoformat(),
                "to": df["created_at"].max().date().isoformat(),
            },
            "language": LANGUAGE,
            "n_reviews_sampled": df.height,
            "effective_n": round(stats.effective_n(w.tolist()), 1),
            "small_sample": df.height < MIN_REVIEWS,
            "unweighted_reviews": int(df["unweighted"].sum()),
            **(
                {
                    "note": f"the window did not cover both sides of split_date; used "
                    f"{args.window_months} months on each side instead"
                }
                if args.window_adjusted
                else {}
            ),
            "positive_share": _r(_wmean(up, w)),
            "classifier_positive_share": _r(_wmean((prob >= 0.5).astype(float), w)),
        }
        if args.split_date:  # before the aspects: the part a long output must not lose
            out["comparison"] = self._compare(df, scored, w, up, args.split_date)
        out["aspects"] = self._aspects(df, scored, w)
        out["method"] = (
            "English review sample reweighted to all reviews (year x rating strata); aspects "
            f"by keyword, aspect polarity by the {self.model_name} on the sentences that "
            "mention the aspect"
        )
        return out

    @staticmethod
    def _compare(
        df: pl.DataFrame, scored: dict[str, Any], w: np.ndarray, up: np.ndarray, split: date
    ) -> dict[str, Any]:
        cut = datetime.combine(split, time.min)
        after = (df["created_at"] >= cut).to_numpy()
        sides = {"before": ~after, "after": after}
        n_eff = {k: stats.effective_n(w[m].tolist()) for k, m in sides.items()}

        def test(values: np.ndarray) -> dict[str, Any]:
            p = {k: _wmean(values[m], w[m]) if m.any() else None for k, m in sides.items()}
            if p["before"] is None or p["after"] is None:
                return {
                    "before": _r(p["before"]),
                    "after": _r(p["after"]),
                    "diff": None,
                    "p_value": None,
                }
            t = stats.two_proportions(
                p["before"] * n_eff["before"],
                n_eff["before"],
                p["after"] * n_eff["after"],
                n_eff["after"],
            )
            return {
                "before": _r(p["before"]),
                "after": _r(p["after"]),
                "diff": _r(t["diff"]),
                "p_value": _r(t["p_value"]),
            }

        return {
            "split_date": split.isoformat(),
            "n_sampled": {k: int(m.sum()) for k, m in sides.items()},
            "effective_n": {k: round(v, 1) for k, v in n_eff.items()},
            "positive_share": test(up),
            # complaint share (negative mentions over all reviews) before vs after
            "complaint_share_by_aspect": {a: test(scored["negative"][a]) for a in ASPECTS},
            "method": "weighted shares, two-proportion z-test on Kish effective sample sizes",
        }
