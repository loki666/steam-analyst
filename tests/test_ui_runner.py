"""The worker-thread runner behind the live timer: events arrive in order, errors come back."""

import threading

from steam_game_analyst.ui.runner import BackgroundRun


def drain(run: BackgroundRun) -> list[str]:
    seen: list[str] = []
    while not run.finished():
        seen += [node for node, _ in run.poll(0.05)]
    return seen


def test_events_arrive_in_order_and_the_result_is_kept():
    release = threading.Event()

    def agent(on_step):
        on_step("router", {"games": []})
        release.wait(5)  # still working: the page keeps ticking meanwhile
        on_step("answer", {})
        return {"answer": "ok"}

    run = BackgroundRun(agent).start()
    assert [n for n, _ in run.poll(5)] == ["router"]
    assert run.poll(0.05) == [] and not run.finished()
    release.set()
    assert drain(run) == ["answer"]
    assert run.result == {"answer": "ok"} and run.error is None


def test_an_error_in_the_worker_is_handed_back():
    def agent(on_step):
        raise ConnectionError("Failed to connect to Ollama")

    run = BackgroundRun(agent).start()
    drain(run)
    assert run.result is None and isinstance(run.error, ConnectionError)
