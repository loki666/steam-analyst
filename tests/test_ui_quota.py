"""Daily question caps for the public demo: per visitor and in total (the hosted LLM's free
allowance is shared by everyone)."""

import datetime
import threading

from steam_game_analyst.ui.quota import DailyQuota, visitor_id


class Clock:
    def __init__(self) -> None:
        self.day = datetime.date(2026, 10, 9)

    def __call__(self) -> datetime.date:
        return self.day


def test_each_visitor_gets_their_own_daily_allowance():
    q = DailyQuota(per_visitor=2, total=None, today=Clock())
    assert [q.take("a").allowed for _ in range(3)] == [True, True, False]
    assert q.take("a").reason == "visitor"
    assert q.take("b").allowed


def test_the_total_cap_stops_everyone():
    q = DailyQuota(per_visitor=None, total=3, today=Clock())
    assert [q.take(v).allowed for v in "abc"] == [True, True, True]
    v = q.take("d")
    assert not v.allowed and v.reason == "total"


def test_a_refused_question_is_not_counted():
    q = DailyQuota(per_visitor=1, total=2, today=Clock())
    assert q.take("a").allowed
    assert not q.take("a").allowed  # refused: must not use up the shared total
    assert q.take("b").allowed


def test_counts_reset_with_the_day():
    clock = Clock()
    q = DailyQuota(per_visitor=1, total=1, today=clock)
    assert q.take("a").allowed and not q.take("a").allowed
    clock.day += datetime.timedelta(days=1)
    assert q.take("a").allowed


def test_no_caps_means_unlimited():
    q = DailyQuota(per_visitor=None, total=None, today=Clock())
    assert all(q.take("a").allowed for _ in range(500))


def test_concurrent_takes_never_exceed_the_total():
    q = DailyQuota(per_visitor=None, total=50, today=Clock())
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(20):
            ok = q.take("x").allowed
            with lock:
                results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 50


def test_visitor_is_the_first_forwarded_address_behind_the_proxy():
    assert visitor_id({"X-Forwarded-For": "203.0.113.7, 172.18.0.4"}, "172.18.0.4") == (
        "203.0.113.7"
    )
    assert visitor_id({}, "198.51.100.2") == "198.51.100.2"
    assert visitor_id({}, None) == "local"


def test_a_non_string_peer_address_counts_as_local():
    assert visitor_id({}, object()) == "local"  # e.g. Streamlit's test harness
