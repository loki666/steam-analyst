"""Plain statistics for the tools (no scipy): proportion tests, a weighted trend line and a
single binomial change point."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def chi2_1df_sf(x: float) -> float:
    """Survival function of chi-square with 1 degree of freedom."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2))


def _t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p for Student's t via the regularised incomplete beta (continued fraction)."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betainc(df / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    # Lentz's continued fraction
    tiny = 1e-30
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    f = d
    for m in range(1, 300):
        for num in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            f *= c * d
        if abs(c * d - 1.0) < 1e-12:
            break
    return front * f / a


def two_proportions(k1: float, n1: float, k2: float, n2: float) -> dict[str, Any]:
    """Pooled two-proportion z-test of p2 - p1 (n may be an effective sample size)."""
    if n1 <= 0 or n2 <= 0:
        return {"p1": None, "p2": None, "diff": None, "z": None, "p_value": None}
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    return {
        "p1": p1,
        "p2": p2,
        "diff": p2 - p1,
        "z": z,
        "p_value": normal_two_sided_p(z) if se > 0 else 1.0,
    }


def effective_n(weights: Sequence[float]) -> float:
    """Kish effective sample size of a weighted sample."""
    s, s2 = sum(weights), sum(w * w for w in weights)
    return s * s / s2 if s2 > 0 else 0.0


def weighted_slope(
    xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]
) -> dict[str, Any] | None:
    """Weighted least squares y = a + b x, with a t-test on b (weights = review counts)."""
    n = len(xs)
    if n < 3 or sum(ws) <= 0:
        return None
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs, strict=True)) / sw
    my = sum(w * y for w, y in zip(ws, ys, strict=True)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs, strict=True))
    if sxx == 0:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    rss = sum(w * (y - intercept - slope * x) ** 2 for w, x, y in zip(ws, xs, ys, strict=True))
    # residual variance per unit weight, scaled back to the mean weight
    sigma2 = rss / (n - 2) if n > 2 else 0.0
    se = math.sqrt(sigma2 / sxx) if sigma2 > 0 else 0.0
    if se == 0:
        p = 0.0 if slope != 0 else 1.0
        t = math.inf if slope != 0 else 0.0
    else:
        t = slope / se
        p = _t_two_sided_p(t, n - 2)
    return {"slope": slope, "intercept": intercept, "se": se, "t": t, "p_value": p, "n_points": n}


def _binom_ll(k: float, n: float) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    out = 0.0
    if 0 < p:
        out += k * math.log(p)
    if p < 1:
        out += (n - k) * math.log(1 - p)
    return out


def binomial_change_point(
    positives: Sequence[float], totals: Sequence[float], min_segment: int = 3
) -> dict[str, Any] | None:
    """Single change point in a binomial rate: the split maximising the likelihood ratio.
    The p-value (chi-square, 1 df) ignores the search over split points, so it is optimistic."""
    m = len(totals)
    if m < 2 * min_segment:
        return None
    k_all, n_all = sum(positives), sum(totals)
    ll0 = _binom_ll(k_all, n_all)
    best: tuple[float, int] | None = None
    for i in range(min_segment, m - min_segment + 1):
        k1, n1 = sum(positives[:i]), sum(totals[:i])
        lr = 2 * (_binom_ll(k1, n1) + _binom_ll(k_all - k1, n_all - n1) - ll0)
        if best is None or lr > best[0]:
            best = (lr, i)
    if best is None:
        return None
    lr, i = best
    k1, n1 = sum(positives[:i]), sum(totals[:i])
    return {
        "index": i,
        "before": k1 / n1 if n1 else None,
        "after": (k_all - k1) / (n_all - n1) if n_all - n1 else None,
        "lr_stat": lr,
        "p_value": chi2_1df_sf(lr),
    }
