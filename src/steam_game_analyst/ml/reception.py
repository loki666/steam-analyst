"""Reception predictor: LightGBM on catalog launch metadata -> positive-review-ratio bucket.

Target: Steam's own review-score boundaries (70% = "Mostly Positive", 80% = "Very Positive"),
with everything under 70% merged because "Negative" (<40%) is ~1% of games. Games need at
least MIN_VOTES votes (Steam's floor for "Very Positive"), so the ratio is not noise.

Only information known at launch is used: price, date, age rating, platforms, languages,
store-page text length, genres, developer-chosen categories, tags (minus evaluative ones such
as "Great Soundtrack" that players add after release) and the developer's / publisher's track
record on games released strictly earlier. Excluded as post-launch: owners, CCU, playtime,
recommendations, Metacritic, DLC count, achievements, tag counts, Trading Cards.

Split by release date (train < 2022 <= validation < 2023 <= test < 2024-06, so every test
game had a year of reviews before the catalog snapshot): the share of very positive
games rises every year, so a random split would overstate accuracy. SHAP values are LightGBM's
TreeSHAP contributions (`pred_contrib`), in log-odds of the predicted bucket."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, log_loss

BUCKETS: tuple[str, ...] = ("negative_or_mixed", "mostly_positive", "very_positive")
BOUNDARIES = (0.70, 0.80)
MIN_VOTES = 50
TRAIN_FROM = date(2014, 1, 1)
VALID_FROM = date(2022, 1, 1)
TEST_FROM = date(2023, 1, 1)
# The FronkonGames catalog has no snapshot date; vote counts collapse for games released after
# May 2025, so it was taken around 2025-06-01. A rating needs time to settle (early reviews
# run high), so only games with at least MATURITY_MONTHS of reviews before the snapshot are
# tested. Train and validation games are all older than that.
CATALOG_SNAPSHOT = date(2025, 6, 1)
MATURITY_MONTHS = 12
TEST_TO = date(CATALOG_SNAPSHOT.year - MATURITY_MONTHS // 12, CATALOG_SNAPSHOT.month, 1)
MIN_TAG_GAMES = 200  # tag / category kept as a feature when this many training games carry it
MODEL_FILE = "reception_lgbm.txt"
SPEC_FILE = "reception_lgbm.spec.json"
METRICS_FILE = "reception_lgbm.metrics.json"
HISTORY_FILE = "reception_track_record.parquet"

# player-applied verdicts, not descriptions: they encode the reception we predict
EVALUATIVE_TAGS = frozenset(
    {
        "Great Soundtrack",
        "Classic",
        "Cult Classic",
        "Masterpiece",
        "Beautiful",
        "Addictive",
        "Epic",
        "Memes",
        "Replay Value",
        "Unforgiving",
        "Nostalgia",
        "Immersive",
        "Good Soundtrack",
        "Underrated",
        "Overrated",
    }
)
# granted or switched on by Valve after release, or platform-wide
POST_LAUNCH_CATEGORIES = frozenset(
    {
        "Steam Trading Cards",
        "Family Sharing",
        # Remote Play was switched on retroactively for most existing games
        "Remote Play Together",
        "Remote Play on TV",
        "Remote Play on Tablet",
        "Remote Play on Phone",
    }
)
SOFTWARE_GENRES = frozenset(
    {
        "Utilities",
        "Design & Illustration",
        "Animation & Modeling",
        "Education",
        "Video Production",
        "Game Development",
        "Audio Production",
        "Software Training",
        "Photo Editing",
        "Web Publishing",
        "Accounting",
        "Movie",
        "Documentary",
        "360 Video",
        "Tutorial",
        "Short",
        "Episodic",
    }
)
_NUMERIC = [
    "price_usd",
    "is_free",
    "release_year",
    "release_month",
    "required_age",
    "n_languages",
    "windows",
    "mac",
    "linux",
    "about_chars",
    "short_description_chars",
    "n_genres",
    "self_published",
    "dev_prior_games",
    "dev_prior_mean_ratio",
    "pub_prior_games",
    "pub_prior_mean_ratio",
]


def bucket_of(ratio: float) -> int:
    return int(sum(ratio >= b for b in BOUNDARIES))


class GameSpec(BaseModel):
    """Launch metadata of a (possibly hypothetical) game. Unknown fields stay missing."""

    price_usd: float = Field(ge=0, le=1000, description="Launch price in USD (0 = free)")
    tags: list[str] = Field(default_factory=list, description="Steam user tags, e.g. Survival")
    genres: list[str] = Field(default_factory=list, description="Steam genres, e.g. Action")
    categories: list[str] = Field(
        default_factory=list, description="Steam categories, e.g. Online Co-op, Single-player"
    )
    release_date: date | None = Field(None, description="Planned release date (default: today)")
    required_age: int | None = Field(None, ge=0, le=21)
    n_languages: int | None = Field(None, ge=1, le=120)
    windows: bool | None = None
    mac: bool | None = None
    linux: bool | None = None
    developer: str | None = None
    publisher: str | None = None


# Store-page fields every catalog game has. Missing in the catalog means a bare listing, so
# for a hypothetical game an unstated value gets the training median, not "missing".
IMPUTE_TYPICAL = (
    "required_age",
    "n_languages",
    "windows",
    "mac",
    "linux",
    "about_chars",
    "short_description_chars",
    "self_published",
)
FEW_TAGS = 5  # catalog games carry ~15 tags; fewer looks like an obscure listing


@dataclass(frozen=True)
class FeatureSpec:
    genres: tuple[str, ...]
    tags: tuple[str, ...]
    categories: tuple[str, ...]
    defaults: dict[str, float] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return (
            _NUMERIC
            + [f"genre: {g}" for g in self.genres]
            + [f"tag: {t}" for t in self.tags]
            + [f"category: {c}" for c in self.categories]
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "genres": self.genres,
            "tags": self.tags,
            "categories": self.categories,
            "defaults": self.defaults,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> FeatureSpec:
        return cls(
            tuple(d["genres"]),
            tuple(d["tags"]),
            tuple(d["categories"]),
            dict(d.get("defaults", {})),
        )


def labelled_games(games_all: pl.DataFrame) -> pl.DataFrame:
    """Released games with enough votes and no software genre, with the bucket label."""
    return games_all.filter(
        pl.col("release_date").is_not_null()
        & ((pl.col("positive") + pl.col("negative")) >= MIN_VOTES)
        & ~pl.col("genres").list.eval(pl.element().is_in(list(SOFTWARE_GENRES))).list.any()
    ).with_columns(
        pl.col("positive_ratio").map_elements(bucket_of, return_dtype=pl.Int64).alias("bucket")
    )


def track_record(games_all: pl.DataFrame) -> pl.DataFrame:
    """name, role (developer|publisher), release_date, ratio for every game with enough votes."""
    voted = games_all.filter(
        pl.col("release_date").is_not_null()
        & ((pl.col("positive") + pl.col("negative")) >= MIN_VOTES)
    )
    frames = [
        voted.select(
            pl.col(col).list.first().str.to_lowercase().alias("name"),
            pl.lit(role).alias("role"),
            "release_date",
            pl.col("positive_ratio").alias("ratio"),
        ).filter(pl.col("name").is_not_null() & (pl.col("name") != ""))
        for col, role in (("developers", "developer"), ("publishers", "publisher"))
    ]
    return pl.concat(frames)


def _prior(history: pl.DataFrame, role: str, name: str | None, before: date) -> tuple[float, float]:
    if not name:
        return 0.0, float("nan")
    h = history.filter(
        (pl.col("role") == role)
        & (pl.col("name") == name.lower())
        & (pl.col("release_date") < before)
    )
    return float(h.height), (float(h["ratio"].mean()) if h.height else float("nan"))


def _with_priors(games: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """Vectorised prior track record: games by the same first developer / publisher released
    strictly before (same-day releases do not see each other)."""
    out = games
    for col, role, prefix in (
        ("developers", "developer", "dev"),
        ("publishers", "publisher", "pub"),
    ):
        h = (
            history.filter(pl.col("role") == role)
            .group_by("name", "release_date")
            .agg(pl.len().alias("_n"), pl.col("ratio").sum().alias("_s"))
            .sort("name", "release_date")
            .with_columns(
                pl.col("_n").cum_sum().shift(1, fill_value=0).over("name").alias("_cn"),
                pl.col("_s").cum_sum().shift(1, fill_value=0.0).over("name").alias("_cs"),
            )
            .select("name", "release_date", "_cn", "_cs")
        )
        out = (
            out.with_columns(pl.col(col).list.first().str.to_lowercase().alias("_name"))
            .sort("release_date")
            .join_asof(
                h.sort("release_date"),
                left_on="release_date",
                right_on="release_date",
                by_left="_name",
                by_right="name",
                strategy="backward",
                check_sortedness=False,  # both sides sorted by date above
                allow_exact_matches=True,  # the row for this date holds counts *before* it
            )
            .with_columns(
                pl.col("_cn").fill_null(0).cast(pl.Float64).alias(f"{prefix}_prior_games"),
                (pl.col("_cs") / pl.col("_cn")).alias(f"{prefix}_prior_mean_ratio"),
            )
            .drop("_name", "_cn", "_cs", "name", strict=False)
        )
    return out


def _vocab(values: pl.Series, min_games: int, exclude: frozenset[str]) -> tuple[str, ...]:
    counts = values.explode(empty_as_null=True).drop_nulls().value_counts()
    keep = counts.filter((pl.col("count") >= min_games) & ~pl.col(values.name).is_in(list(exclude)))
    return tuple(sorted(keep[values.name].to_list()))


def build_spec(train: pl.DataFrame) -> FeatureSpec:
    spec = FeatureSpec(
        genres=_vocab(train["genres"], MIN_TAG_GAMES, SOFTWARE_GENRES),
        tags=_vocab(train["tags"], MIN_TAG_GAMES, EVALUATIVE_TAGS),
        categories=_vocab(train["categories"], MIN_TAG_GAMES, POST_LAUNCH_CATEGORIES),
    )
    x = feature_matrix(train, spec)
    defaults = {name: float(np.nanmedian(x[:, spec.names.index(name)])) for name in IMPUTE_TYPICAL}
    return FeatureSpec(spec.genres, spec.tags, spec.categories, defaults)


def _multi_hot(lists: list[list[str] | None], vocab: tuple[str, ...]) -> np.ndarray:
    index = {v: i for i, v in enumerate(vocab)}
    m = np.zeros((len(lists), len(vocab)), dtype=np.float32)
    for r, items in enumerate(lists):
        for it in items or []:
            if it in index:
                m[r, index[it]] = 1.0
    return m


def feature_matrix(games: pl.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """Rows of `games` (with prior columns) -> feature matrix in `spec.names` order."""
    d = games.with_columns(
        (pl.col("price_usd").fill_null(0) == 0).cast(pl.Float64).alias("is_free"),
        pl.col("release_date").dt.year().cast(pl.Float64).alias("release_year"),
        pl.col("release_date").dt.month().cast(pl.Float64).alias("release_month"),
        pl.col("supported_languages").list.len().cast(pl.Float64).alias("n_languages"),
        pl.col("about_the_game").str.len_chars().cast(pl.Float64).alias("about_chars"),
        pl.col("short_description")
        .str.len_chars()
        .cast(pl.Float64)
        .alias("short_description_chars"),
        pl.col("genres").list.len().cast(pl.Float64).alias("n_genres"),
        (pl.col("developers").list.first() == pl.col("publishers").list.first())
        .cast(pl.Float64)
        .alias("self_published"),
    )
    numeric = d.select([pl.col(c).cast(pl.Float64) for c in _NUMERIC]).to_numpy()
    return np.hstack(
        [
            numeric,
            _multi_hot(d["genres"].to_list(), spec.genres),
            _multi_hot(d["tags"].to_list(), spec.tags),
            _multi_hot(d["categories"].to_list(), spec.categories),
        ]
    ).astype(np.float64)


PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": len(BUCKETS),
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    # explicit: LightGBM's default (all cores) stalls on OpenMP next to polars' thread pool
    "num_threads": min(8, os.cpu_count() or 1),
}


def _ece(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    """Top-label expected calibration error + the reliability table behind it."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += m.mean() * gap
        table.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(m.sum()),
                "mean_confidence": float(conf[m].mean()),
                "accuracy": float(correct[m].mean()),
            }
        )
    return float(ece), table


def _brier(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[y]
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def _scores(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = proba.argmax(axis=1)
    ece, table = _ece(y, proba)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-9, 1), labels=list(range(len(BUCKETS))))),
        "brier": _brier(y, proba),
        "ece": ece,
        "reliability": table,
    }


def _bootstrap_ci(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0
) -> dict[str, list[float]]:
    """95% CIs for accuracy of predictions a, and for acc(a) - acc(b), resampling games."""
    rng = np.random.default_rng(seed)
    acc, diff = [], []
    ok_a, ok_b = (a == y).astype(float), (b == y).astype(float)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        acc.append(ok_a[i].mean())
        diff.append(ok_a[i].mean() - ok_b[i].mean())
    q = [2.5, 97.5]
    return {
        "accuracy_ci95": [float(v) for v in np.percentile(acc, q)],
        "accuracy_gain_vs_majority_ci95": [float(v) for v in np.percentile(diff, q)],
    }


def split(labelled: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    d = pl.col("release_date")
    return (
        labelled.filter((d >= TRAIN_FROM) & (d < VALID_FROM)),
        labelled.filter((d >= VALID_FROM) & (d < TEST_FROM)),
        labelled.filter((d >= TEST_FROM) & (d < TEST_TO)),
    )


def train(games_all: pl.DataFrame, seed: int = 13) -> tuple[ReceptionModel, dict[str, Any]]:
    history = track_record(games_all)
    data = _with_priors(labelled_games(games_all), history)
    tr, va, te = split(data)
    spec = build_spec(tr)
    xs = {k: feature_matrix(d, spec) for k, d in (("train", tr), ("valid", va), ("test", te))}
    ys = {k: d["bucket"].to_numpy() for k, d in (("train", tr), ("valid", va), ("test", te))}
    booster = lgb.train(
        {**PARAMS, "seed": seed},
        lgb.Dataset(xs["train"], ys["train"]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(xs["valid"], ys["valid"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    model = ReceptionModel(booster, spec, history)
    proba = model.predict_matrix(xs["test"])
    prior = np.bincount(ys["train"], minlength=len(BUCKETS)) / len(ys["train"])
    majority = int(prior.argmax())
    base_proba = np.tile(prior, (len(ys["test"]), 1))
    contrib = model.contributions(xs["test"])  # (n, features, classes)
    mean_abs = np.abs(contrib).mean(axis=(0, 2))
    top = np.argsort(-mean_abs)[:20]
    metrics = {
        "model": "LightGBM multiclass",
        "buckets": {b: f">= {lo:.0%}" for b, lo in zip(BUCKETS, (0.0, *BOUNDARIES), strict=True)},
        "min_votes": MIN_VOTES,
        "split": {
            "train": f"{TRAIN_FROM} .. {VALID_FROM}",
            "valid": f"{VALID_FROM} .. {TEST_FROM} (early stopping)",
            "test": f"{TEST_FROM} .. {TEST_TO} (>= {MATURITY_MONTHS} months of reviews "
            f"before the ~{CATALOG_SNAPSHOT} catalog snapshot)",
        },
        "n": {k: int(len(v)) for k, v in ys.items()},
        "class_share": {
            k: [float(x) for x in np.bincount(v, minlength=len(BUCKETS)) / len(v)]
            for k, v in ys.items()
        },
        "best_iteration": int(booster.best_iteration),
        "n_features": len(spec.names),
        "test": {
            **_scores(ys["test"], proba),
            **_bootstrap_ci(ys["test"], proba.argmax(axis=1), np.full(len(ys["test"]), majority)),
        },
        "majority_baseline": {
            **_scores(ys["test"], base_proba),
            "predicts": BUCKETS[majority],
            "note": "train class frequencies as probabilities; argmax = majority class",
        },
        "shap_mean_abs_top20": {spec.names[i]: float(mean_abs[i]) for i in top},
        "caveats": [
            "price, tags, categories, platforms, languages and store text are the catalog "
            "snapshot, not the launch state (price cuts, free-to-play switches and later "
            "tags leak some post-launch information), so test scores may be optimistic",
            f"only games with >= {MIN_VOTES} votes are modelled: games that never found "
            "players are outside the target population",
            "release_year cannot extrapolate: later years behave like the last training year",
        ],
    }
    return model, metrics


class ReceptionModel:
    def __init__(self, booster: lgb.Booster, spec: FeatureSpec, history: pl.DataFrame) -> None:
        self.booster = booster
        self.spec = spec
        self.history = history

    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster.predict(x, num_iteration=self.booster.best_iteration))

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """TreeSHAP values, shape (rows, features, classes); the bias term is dropped."""
        raw = np.asarray(
            self.booster.predict(x, num_iteration=self.booster.best_iteration, pred_contrib=True)
        )
        k, f = len(BUCKETS), len(self.spec.names)
        return raw.reshape(len(x), k, f + 1)[:, :, :f].transpose(0, 2, 1)

    def _canonical(self, values: list[str], vocab: tuple[str, ...]) -> tuple[list[str], list[str]]:
        """Match user-typed names to the vocabulary ignoring case and punctuation
        ("coop" -> "Co-op"); returns (matched, unknown)."""

        def norm(s: str) -> str:
            return "".join(ch for ch in s.casefold() if ch.isalnum())

        lookup = {norm(v): v for v in vocab}
        matched, unknown = [], []
        for v in values:
            hit = lookup.get(norm(v))
            (matched if hit else unknown).append(hit or v)
        return list(dict.fromkeys(matched)), unknown

    def frame(self, spec: GameSpec, today: date | None = None) -> tuple[pl.DataFrame, dict]:
        released = spec.release_date or today or date.today()
        tags, bad_tags = self._canonical(spec.tags, self.spec.tags)
        genres, bad_genres = self._canonical(spec.genres, self.spec.genres)
        cats, bad_cats = self._canonical(spec.categories, self.spec.categories)
        dev_n, dev_r = _prior(self.history, "developer", spec.developer, released)
        pub_n, pub_r = _prior(self.history, "publisher", spec.publisher, released)
        row = {
            "price_usd": spec.price_usd,
            "release_date": released,
            "required_age": spec.required_age,
            "supported_languages": None if spec.n_languages is None else ["?"] * spec.n_languages,
            "windows": spec.windows,
            "mac": spec.mac,
            "linux": spec.linux,
            "about_the_game": None,
            "short_description": None,
            "genres": genres,
            "categories": cats,
            "tags": tags,
            "developers": [spec.developer] if spec.developer else None,
            "publishers": [spec.publisher] if spec.publisher else None,
            "dev_prior_games": dev_n,
            "dev_prior_mean_ratio": dev_r,
            "pub_prior_games": pub_n,
            "pub_prior_mean_ratio": pub_r,
        }
        schema = {
            "price_usd": pl.Float64,
            "release_date": pl.Date,
            "required_age": pl.Int64,
            "supported_languages": pl.List(pl.Utf8),
            "windows": pl.Boolean,
            "mac": pl.Boolean,
            "linux": pl.Boolean,
            "about_the_game": pl.Utf8,
            "short_description": pl.Utf8,
            "genres": pl.List(pl.Utf8),
            "categories": pl.List(pl.Utf8),
            "tags": pl.List(pl.Utf8),
            "developers": pl.List(pl.Utf8),
            "publishers": pl.List(pl.Utf8),
            "dev_prior_games": pl.Float64,
            "dev_prior_mean_ratio": pl.Float64,
            "pub_prior_games": pl.Float64,
            "pub_prior_mean_ratio": pl.Float64,
        }
        used = {
            "release_date": released.isoformat(),
            "tags": tags,
            "genres": genres,
            "categories": cats,
            "unknown": {"tags": bad_tags, "genres": bad_genres, "categories": bad_cats},
            "developer_prior_games": int(dev_n),
            "publisher_prior_games": int(pub_n),
        }
        return pl.DataFrame([row], schema=schema), used

    def predict(self, spec: GameSpec, top_k: int = 8, today: date | None = None) -> dict:
        frame, used = self.frame(spec, today)
        x = feature_matrix(frame, self.spec)
        imputed = {}
        for name, value in self.spec.defaults.items():
            i = self.spec.names.index(name)
            if np.isnan(x[0, i]):
                x[0, i] = value
                imputed[name] = value
        used["imputed_typical"] = imputed
        warnings = []
        if len(used["tags"]) < FEW_TAGS:
            warnings.append(
                f"only {len(used['tags'])} known tags: catalog games carry ~15, and sparse tag "
                "lists belong to obscure listings, so the prediction leans negative; pass more "
                "tags for a fairer estimate"
            )
        if imputed:
            warnings.append(f"typical catalog values assumed for {sorted(imputed)}")
        proba = self.predict_matrix(x)[0]
        k = int(proba.argmax())
        contrib = self.contributions(x)[0, :, k]
        order = np.argsort(-np.abs(contrib))
        top = [
            {
                "feature": self.spec.names[i],
                "value": None if np.isnan(x[0, i]) else float(x[0, i]),
                "shap": round(float(contrib[i]), 4),
                "pushes": "towards" if contrib[i] > 0 else "away from",
            }
            for i in order[:top_k]
            if contrib[i] != 0
        ]
        return {
            "bucket": BUCKETS[k],
            "probabilities": {b: round(float(p), 4) for b, p in zip(BUCKETS, proba, strict=True)},
            "top_features": top,
            "inputs_used": used,
            "warnings": warnings,
        }


def save(model: ReceptionModel, metrics: dict[str, Any], models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MODEL_FILE
    model.booster.save_model(str(path), num_iteration=model.booster.best_iteration)
    (models_dir / SPEC_FILE).write_text(json.dumps(model.spec.to_json(), indent=1))
    (models_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=1))
    model.history.write_parquet(models_dir / HISTORY_FILE)
    return path


def load(models_dir: Path) -> ReceptionModel:
    path = models_dir / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `sga train reception`")
    booster = lgb.Booster(model_file=str(path))
    spec = FeatureSpec.from_json(json.loads((models_dir / SPEC_FILE).read_text()))
    return ReceptionModel(booster, spec, pl.read_parquet(models_dir / HISTORY_FILE))
