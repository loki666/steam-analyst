"""review_trend: monthly (or quarterly) review volume and positive share for one game, a
weighted trend line, and either a single change point or a before/after test around a date.

Reads review_monthly, which counts every cleaned review in the Mendeley files (the population),
not the stratified text sample. "Last N months" is relative to the game's last complete month
in the data, not to today: the review corpus ends in December 2024."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from steam_game_analyst.tools import stats
from steam_game_analyst.tools.catalog import CatalogDB

SIGNIFICANCE = 0.05


class TrendArgs(BaseModel):
    appid: int = Field(description="Steam appid of the game")
    granularity: Literal["month", "quarter"] = "month"
    last_n_months: int | None = Field(
        None, ge=1, le=120, description="Only the last N months of data (relative to data end)"
    )
    date_from: date | None = Field(None, description="Start date (inclusive)")
    date_to: date | None = Field(None, description="End date (inclusive)")
    event_date: date | None = Field(
        None, description="Date of a patch or event: compare the windows before and after it"
    )
    window_months: int = Field(6, ge=1, le=36, description="Months on each side of event_date")


def month_add(d: date, months: int) -> date:
    idx = d.year * 12 + d.month - 1 + months
    return date(idx // 12, idx % 12 + 1, 1)


def _quarter(d: date) -> date:
    return date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)


def aggregate(rows: list[tuple[date, int, int]], granularity: str) -> list[tuple[date, int, int]]:
    if granularity == "month":
        return rows
    out: dict[date, list[int]] = {}
    for m, n, k in rows:
        acc = out.setdefault(_quarter(m), [0, 0])
        acc[0] += n
        acc[1] += k
    return [(q, n, k) for q, (n, k) in sorted(out.items())]


def densify(rows: list[tuple[date, int, int]], lo: date, hi: date) -> list[tuple[date, int, int]]:
    """Every month from lo to hi, with (month, 0, 0) where the game had no reviews."""
    have = {m: (m, n, k) for m, n, k in rows}
    out, m = [], lo
    while m <= hi:
        out.append(have.get(m, (m, 0, 0)))
        m = month_add(m, 1)
    return out


def _share(n: int, k: int) -> float | None:
    return round(k / n, 4) if n else None


class ReviewTrend:
    def __init__(self, db: CatalogDB) -> None:
        self.db = db
        row = db.fetch("SELECT max(created_at)::DATE FROM reviews")
        self.data_end: date | None = row[0][0] if row else None
        # last complete month of the corpus: a month the scrape stopped in is partial
        if self.data_end is None:
            self.corpus_last_month: date | None = None
        elif self.data_end.day < 28:
            self.corpus_last_month = month_add(self.data_end.replace(day=1), -1)
        else:
            self.corpus_last_month = self.data_end.replace(day=1)

    def _catalog(self, appid: int) -> dict[str, Any] | None:
        rows = self.db.fetch(
            "SELECT name, release_date, positive, negative, positive_ratio FROM games_all "
            "WHERE appid = ?",
            [appid],
        )
        if not rows:
            return None
        name, rel, pos, neg, ratio = rows[0]
        return {
            "name": name,
            "release_date": rel.isoformat() if rel else None,
            "positive_votes": pos,
            "negative_votes": neg,
            "positive_ratio_all_time": round(ratio, 4) if ratio is not None else None,
        }

    def run(self, args: TrendArgs) -> dict[str, Any]:
        rows: list[tuple[date, int, int]] = [
            (m, int(n), int(k))
            for m, n, k in self.db.fetch(
                "SELECT month, n_reviews, n_positive FROM review_monthly WHERE appid = ? "
                "ORDER BY month",
                [args.appid],
            )
        ]
        names = self.db.fetch("SELECT name FROM games WHERE appid = ?", [args.appid])
        if not rows:
            catalog = self._catalog(args.appid)
            return {
                "error": f"no review time series for appid {args.appid}: the review corpus "
                "(Mendeley, 2020-2024 sample of Steam reviews) has no reviews for this game",
                "catalog": catalog,
                "note": "catalog vote counts are all-time totals at the snapshot date; "
                "they cannot show a trend",
            }
        last = self.corpus_last_month or rows[-1][0]
        complete = [r for r in rows if r[0] <= last]
        dropped = rows[-1][0] if len(complete) < len(rows) else None
        hi = min(args.date_to.replace(day=1), last) if args.date_to else last
        lo = args.date_from.replace(day=1) if args.date_from else (complete or rows)[0][0]
        if args.last_n_months:
            lo = max(lo, month_add(hi, -(args.last_n_months - 1)))
        elif args.event_date and not (args.date_from or args.date_to):
            # default view around an event: the two comparison windows and the event month
            start = args.event_date.replace(day=1)
            lo = month_add(start, -args.window_months)
            hi = min(hi, month_add(start, args.window_months))
        window = densify([r for r in complete if lo <= r[0] <= hi], lo, hi)
        series = aggregate(window, args.granularity)
        out: dict[str, Any] = {
            "game": {"appid": args.appid, "name": names[0][0] if names else None},
            "granularity": args.granularity,
            "data_through": last.isoformat(),
            "dropped_partial_month": dropped.isoformat() if dropped else None,
            "window": {
                "from": window[0][0].isoformat() if window else None,
                "to": window[-1][0].isoformat() if window else None,
            },
            "series": [
                {"period": p.isoformat(), "n_reviews": n, "positive_share": _share(n, k)}
                for p, n, k in series
            ],
            "source": "review_monthly: every cleaned review in the corpus (population counts)",
        }
        if complete and complete[-1][0] < hi:
            out["note"] = (
                f"this game has no reviews in the corpus after {complete[-1][0].isoformat()}; "
                "later months count as 0"
            )
        n_tot, k_tot = sum(r[1] for r in window), sum(r[2] for r in window)
        out["summary"] = {"n_reviews": n_tot, "positive_share": _share(n_tot, k_tot)}
        out["trend"] = self._trend(window)
        if args.event_date:
            out["event_test"] = self._event(complete, args.event_date, args.window_months)
        else:
            out["change_point"] = self._change_point(series)
        return out

    @staticmethod
    def _trend(window: list[tuple[date, int, int]]) -> dict[str, Any]:
        # x = calendar month index, so gaps between months with reviews keep their length
        pts = [(m.year * 12 + m.month, k / n, n) for m, n, k in window if n]
        fit = stats.weighted_slope([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
        if fit is None:
            return {"direction": "unknown", "note": "fewer than 3 months with reviews"}
        sig = fit["p_value"] < SIGNIFICANCE
        direction = ("up" if fit["slope"] > 0 else "down") if sig else "flat"
        half = len(window) // 2
        first, second = window[:half], window[half:]
        return {
            "direction": direction,
            "positive_share_change_per_year": round(fit["slope"] * 12, 4),
            "p_value": round(fit["p_value"], 4),
            "method": "weighted least squares of monthly positive share on time "
            "(weights = reviews per month)",
            "volume_first_half_avg": round(sum(r[1] for r in first) / max(len(first), 1), 1),
            "volume_second_half_avg": round(sum(r[1] for r in second) / max(len(second), 1), 1),
        }

    @staticmethod
    def _change_point(series: list[tuple[date, int, int]]) -> dict[str, Any] | None:
        cp = stats.binomial_change_point([k for _, _, k in series], [n for _, n, _ in series])
        if cp is None:
            return None
        return {
            "period": series[cp["index"]][0].isoformat(),
            "positive_share_before": round(cp["before"], 4) if cp["before"] is not None else None,
            "positive_share_after": round(cp["after"], 4) if cp["after"] is not None else None,
            "p_value": cp["p_value"],
            "note": "most likely single shift in positive share; the p-value ignores the "
            "search over periods, so treat it as optimistic",
        }

    @staticmethod
    def _event(rows: list[tuple[date, int, int]], event: date, months: int) -> dict[str, Any]:
        start = event.replace(day=1)
        # the event month itself is mixed: it is left out of both windows
        before = [r for r in rows if month_add(start, -months) <= r[0] < start]
        after = [r for r in rows if month_add(start, 1) <= r[0] <= month_add(start, months)]

        def side(rs: list[tuple[date, int, int]]) -> dict[str, Any]:
            n, k = sum(r[1] for r in rs), sum(r[2] for r in rs)
            return {
                "from": rs[0][0].isoformat() if rs else None,
                "to": month_add(rs[-1][0], 1).replace(day=1).isoformat() if rs else None,
                "n_reviews": n,
                "positive_share": _share(n, k),
            }

        b, a = side(before), side(after)
        b["to"] = start.isoformat() if before else None  # exclusive bound
        t = stats.two_proportions(
            sum(r[2] for r in before),
            sum(r[1] for r in before),
            sum(r[2] for r in after),
            sum(r[1] for r in after),
        )
        return {
            "event_date": event.isoformat(),
            "window_months": months,
            "before": b,
            "after": a,
            "diff": round(t["diff"], 4) if t["diff"] is not None else None,
            "p_value": t["p_value"],
            "method": "two-proportion z-test; the event month is excluded from both windows",
        }
