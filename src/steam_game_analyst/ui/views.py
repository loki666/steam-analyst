"""Card copy for tool outputs: titles, meta lines and metric tiles. Pure functions over the
parsed tool output; every number shown comes from that output."""

from __future__ import annotations

from datetime import date
from typing import Any, NamedTuple

from steam_game_analyst.ui.results import pct

MINUS = "−"  # a true minus sign, not a hyphen
SMALL_EFFECTIVE_N = 30  # below this a before/after difference under ~10 points is noise


class Tile(NamedTuple):
    value: str
    caption: str
    help: str | None = None  # the detail, shown on hover


def signed(x: float) -> str:
    """Two significant figures with an explicit sign: +13, −0.06."""
    return f"{MINUS if x < 0 else '+'}{abs(x):.2g}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


BUCKET_LABEL = {
    "negative_or_mixed": "Mixed or negative (< 70%)",
    "mostly_positive": "Mostly positive (70–80%)",
    "very_positive": "Very positive (≥ 80%)",
}


def month_label(iso: str | None) -> str:
    if not iso:
        return "?"
    return date.fromisoformat(iso[:10]).strftime("%b %Y")


def day_label(iso: str | None) -> str:
    if not iso:
        return "?"
    d = date.fromisoformat(iso[:10])
    return f"{d.day} {d.strftime('%b %Y')}"


def fmt_k(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x / 1000:.1f}k" if abs(x) >= 1000 else f"{x:.0f}"


def p_label(p: float | None) -> str:
    if p is None:
        return "p n/a"
    return "p < 0.001" if p < 0.001 else f"p {p:.2g}"


def game_name(out: dict[str, Any]) -> str:
    game = out.get("game") or {}
    return str(game.get("name") or f"appid {game.get('appid', '?')}")


# ------------------------------------------------------------------ review_trend
def trend_title(out: dict[str, Any]) -> str:
    return f"Positive share by {out.get('granularity', 'month')} · {game_name(out)}"


def trend_meta(out: dict[str, Any]) -> str:
    w, s = out.get("window", {}), out.get("summary", {})
    total = s.get("n_reviews")
    tail = f" · every review, {total:,} in total" if total is not None else ""
    return f"{month_label(w.get('from'))} – {month_label(w.get('to'))}{tail}"


def _shares(before: float | None, after: float | None) -> str:
    return f"{pct(before, 0)} → {pct(after, 0)}"


def _shift_tile(out: dict[str, Any]) -> Tile | None:
    if ev := out.get("event_test"):
        b, a = ev.get("before", {}), ev.get("after", {})
        return Tile(
            _shares(b.get("positive_share"), a.get("positive_share")),
            f"before → after · {p_label(ev.get('p_value'))}",
            f"Positive share in the {ev.get('window_months', '?')} months before vs after "
            f"{ev.get('event_date')} ({b.get('n_reviews', '?')} vs {a.get('n_reviews', '?')} "
            f"reviews). {ev.get('method', '')}",
        )
    if cp := out.get("change_point"):
        return Tile(
            _shares(cp.get("positive_share_before"), cp.get("positive_share_after")),
            f"shift · {month_label(cp.get('period'))}",
            f"Most likely single change in positive share. {cp.get('note', '')}",
        )
    return None


def trend_tiles(out: dict[str, Any]) -> list[Tile]:
    t = out.get("trend") or {}
    tiles: list[Tile] = []
    if t.get("direction"):
        per_year = t.get("positive_share_change_per_year")
        slope = f"{signed(100 * per_year)} pts/yr · " if per_year is not None else ""
        tiles.append(Tile(t["direction"], f"{slope}{p_label(t.get('p_value'))}", t.get("method")))
    if shift := _shift_tile(out):
        tiles.append(shift)
    if "volume_first_half_avg" in t:
        first, second = t["volume_first_half_avg"], t["volume_second_half_avg"]
        period = out.get("granularity", "month")
        tiles.append(
            Tile(
                f"{fmt_k(first)} → {fmt_k(second)}",
                f"reviews/{period}, H1 → H2",
                f"Average reviews per {period}: {first:,.0f} in the first half of the window, "
                f"{second:,.0f} in the second.",
            )
        )
    return tiles


def catalog_tiles(out: dict[str, Any]) -> list[Tile]:
    """review_trend for a game without dated reviews: the all-time catalog facts."""
    cat = (out.get("available") or {}).get("all_time_catalog") or {}
    tiles = [Tile(pct(cat.get("positive_ratio_all_time")), "positive, all-time votes")]
    if cat.get("positive_votes") is not None:
        votes = cat["positive_votes"] + (cat.get("negative_votes") or 0)
        tiles.append(Tile(f"{votes:,}", "votes on Steam"))
    if cat.get("game_released_on"):
        tiles.append(Tile(day_label(cat["game_released_on"]), "released"))
    return tiles


# ------------------------------------------------------------------ review_sentiment
def sentiment_meta(out: dict[str, Any]) -> str:
    w = out.get("window", {})
    return (
        f"{day_label(w.get('from'))} – {day_label(w.get('to'))} · "
        f"{out.get('n_reviews_sampled', '?')} sampled reviews · "
        f"effective n {out.get('effective_n', '?')}"
    )


def top_complaint(aspects: dict[str, Any]) -> tuple[str, float] | None:
    scored = [
        (k, v["complaint_share"])
        for k, v in aspects.items()
        if isinstance(v, dict) and v.get("complaint_share") is not None
    ]
    return max(scored, key=lambda kv: kv[1]) if scored else None


def sentiment_tiles(out: dict[str, Any]) -> list[Tile]:
    tiles = [
        Tile(
            pct(out.get("positive_share")),
            "positive (Steam votes)",
            "Share of reviews voted up, reweighted to all reviews by year and rating.",
        ),
        Tile(
            pct(out.get("classifier_positive_share")),
            "positive (classifier)",
            "Share of reviews the text classifier scores positive, same weights.",
        ),
    ]
    if top := top_complaint(out.get("aspects") or {}):
        tiles.append(
            Tile(
                pct(top[1]),
                f"complain about {top[0]}",
                "Share of all reviews with a negative sentence about the aspect "
                "complained about most.",
            )
        )
    return tiles


def comparison_tiles(cmp: dict[str, Any]) -> list[Tile]:
    pos = cmp.get("positive_share") or {}
    eff = cmp.get("effective_n") or {}
    return [
        Tile(
            _shares(pos.get("before"), pos.get("after")),
            f"positive · {p_label(pos.get('p_value'))}",
            cmp.get("method"),
        ),
        Tile(
            f"{eff.get('before', '?')} / {eff.get('after', '?')}",
            "effective n before / after",
            "Kish effective sample size of the weighted review sample on each side.",
        ),
    ]


def comparison_rows(cmp: dict[str, Any]) -> list[tuple[str, float, float, float | None]]:
    """(aspect, before, after, p) for the dumbbell chart, without the aspects that have no
    complaints on either side (`quiet_aspects`)."""
    rows = [
        (k, v["before"], v["after"], v.get("p_value"))
        for k, v in (cmp.get("complaint_share_by_aspect") or {}).items()
        if v.get("before") is not None and v.get("after") is not None
    ]
    return [r for r in rows if r[1] or r[2]]


def quiet_aspects(cmp: dict[str, Any]) -> list[str]:
    return [
        k
        for k, v in (cmp.get("complaint_share_by_aspect") or {}).items()
        if v.get("before") == 0 and v.get("after") == 0
    ]


def small_sides(cmp: dict[str, Any]) -> list[tuple[str, float]]:
    """(side, effective n) for each side of a before/after split below SMALL_EFFECTIVE_N."""
    eff = cmp.get("effective_n") or {}
    return [
        (side, float(eff[side]))
        for side in ("before", "after")
        if eff.get(side) is not None and eff[side] < SMALL_EFFECTIVE_N
    ]


# ------------------------------------------------------------------ predict_reception
def reception_meta(args: dict[str, Any], out: dict[str, Any]) -> str:
    used = out.get("inputs_used") or {}
    parts = []
    if tags := used.get("tags"):
        parts.append("tags: " + ", ".join(tags))
    if args.get("price_usd") is not None:
        parts.append(f"${args['price_usd']:.2f}")
    if used.get("release_date"):
        parts.append(str(used["release_date"]))
    return " · ".join(parts)


def feature_label(f: dict[str, Any]) -> str:
    """A SHAP feature in words: "category: Steam Cloud" with value 0 -> "No Steam Cloud"."""
    name, value = str(f.get("feature", "?")), f.get("value")
    if name == "price_usd" and value is not None:
        return f"Price ${value:.2f}"
    if name == "release_year" and value is not None:
        return f"Release year {value:.0f}"
    if name in ("dev_prior_mean_ratio", "pub_prior_mean_ratio") and value is None:
        who = "Developer" if name.startswith("dev") else "Publisher"
        return f"{who} track record unknown"
    if name in ("mac", "linux", "windows") and value is not None:
        os_name = {"mac": "Mac", "linux": "Linux", "windows": "Windows"}[name]
        return f"{os_name} version" if value else f"No {os_name} version"
    if ":" in name and value in (0.0, 1.0):
        kind, label = (s.strip() for s in name.split(":", 1))
        return f"{kind.capitalize()}: {label}" if value else f"No {label}"
    shown = "unknown" if value is None else f"{value:g}"
    return f"{name.replace('_', ' ')} = {shown}"
