"""Card copy for tool outputs: titles, meta lines and metric tiles. Pure functions over the
parsed tool output; every number shown comes from that output."""

from __future__ import annotations

from datetime import date
from typing import Any

from steam_game_analyst.ui.results import pct

Tile = tuple[str, str]  # (value, caption)

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


def _shift_tile(out: dict[str, Any]) -> Tile | None:
    if ev := out.get("event_test"):
        b, a = ev.get("before", {}), ev.get("after", {})
        value = f"{pct(b.get('positive_share'), 0)} → {pct(a.get('positive_share'), 0)}"
        return value, f"positive before / after event, {p_label(ev.get('p_value'))}"
    if cp := out.get("change_point"):
        value = (
            f"{pct(cp.get('positive_share_before'), 0)} → {pct(cp.get('positive_share_after'), 0)}"
        )
        return value, f"most likely shift, {month_label(cp.get('period'))}"
    return None


def trend_tiles(out: dict[str, Any]) -> list[Tile]:
    t = out.get("trend") or {}
    tiles: list[Tile] = []
    if t.get("direction"):
        per_year = t.get("positive_share_change_per_year")
        slope = f" · {100 * per_year:+.1f} pts/year" if per_year is not None else ""
        tiles.append((t["direction"], f"trend{slope}, {p_label(t.get('p_value'))}"))
    if shift := _shift_tile(out):
        tiles.append(shift)
    if "volume_first_half_avg" in t:
        value = f"{fmt_k(t['volume_first_half_avg'])} → {fmt_k(t['volume_second_half_avg'])}"
        tiles.append((value, "reviews / period, 1st vs 2nd half"))
    return tiles


def catalog_tiles(out: dict[str, Any]) -> list[Tile]:
    """review_trend for a game without dated reviews: the all-time catalog facts."""
    cat = (out.get("available") or {}).get("all_time_catalog") or {}
    tiles: list[Tile] = [(pct(cat.get("positive_ratio_all_time")), "positive, all-time votes")]
    if cat.get("positive_votes") is not None:
        votes = cat["positive_votes"] + (cat.get("negative_votes") or 0)
        tiles.append((f"{votes:,}", "votes on Steam"))
    if cat.get("game_released_on"):
        tiles.append((day_label(cat["game_released_on"]), "released"))
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
    tiles: list[Tile] = [
        (pct(out.get("positive_share")), "positive (Steam votes)"),
        (pct(out.get("classifier_positive_share")), "positive (classifier)"),
    ]
    if top := top_complaint(out.get("aspects") or {}):
        tiles.append((pct(top[1]), f"complain about {top[0]}"))
    return tiles


def comparison_tiles(cmp: dict[str, Any]) -> list[Tile]:
    pos = cmp.get("positive_share") or {}
    eff = cmp.get("effective_n") or {}
    return [
        (
            f"{pct(pos.get('before'), 0)} → {pct(pos.get('after'), 0)}",
            f"positive reviews · {p_label(pos.get('p_value'))}",
        ),
        (f"{eff.get('before', '?')} / {eff.get('after', '?')}", "effective reviews before / after"),
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
