"""Activity feed ("bot chat"): ring-buffer semantics + /live delivery."""
from aiohttp.test_utils import TestClient, TestServer

from app.dashboard.feed import ActivityFeed
from app.dashboard.server import build_app
from app.db.database import Database


def test_feed_tail_after_and_bound():
    f = ActivityFeed(maxlen=5)
    for i in range(8):
        f.say("learn", f"msg {i}")
    assert f.top == 8
    assert [e["id"] for e in f.tail()] == [4, 5, 6, 7, 8]   # bounded at 5
    evs = f.tail(after=6)
    assert [e["text"] for e in evs] == ["msg 6", "msg 7"]
    assert all(e["kind"] == "learn" for e in evs)
    assert f.tail(after=99) == []


async def test_live_includes_new_events(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_PASS", raising=False)
    db = Database(str(tmp_path / "t.db"))
    await db.connect()
    feed = ActivityFeed()
    feed.say("sys", "hello")
    client = TestClient(TestServer(build_app(db, 10000.0, feed=feed)))
    await client.start_server()
    try:
        p = await (await client.get("/live")).json()
        assert p["events_top"] == 1
        assert p["events"][0]["text"] == "hello"

        feed.say("open", "opened BTC")
        p = await (await client.get("/live?ev_after=1")).json()
        assert [e["text"] for e in p["events"]] == ["opened BTC"]

        # a bad ev_after must fall back to "everything", never 500
        p = await (await client.get("/live?ev_after=notanint")).json()
        assert len(p["events"]) == 2
    finally:
        await client.close()
        await db.close()
