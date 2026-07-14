"""Dashboard server: aiohttp JSON API + single-page command deck UI.

Endpoints:
  /            HTML dashboard (app/dashboard/index.html)
  /metrics     performance metrics + system info (mode, exchange, symbols)
  /positions   open paper trades
  /trades      recent closed trades (newest first)
  /analyses    recent analyses (newest first)
  /equity      equity curve

Optional access control: set DASHBOARD_PASS (and optionally DASHBOARD_USER,
default "admin") to require HTTP Basic Auth on every route. Strongly
recommended once live trading is armed, since this server exposes positions,
equity and the live-mode flag on a public URL. If DASHBOARD_PASS is unset the
dashboard is open (a warning is logged at startup).
"""
from __future__ import annotations

import asyncio
import base64
import csv
import hmac
import io
import json
import os
import time
from pathlib import Path

from aiohttp import web

from app.analysis.walkforward import run_walk_forward
from app.dashboard.metrics import compute_metrics
from app.db.database import Database
from app.logging_setup import get_logger

log = get_logger("dashboard")
INDEX_PATH = Path(__file__).with_name("index.html")

_TRADE_COLS = (
    "id", "symbol", "direction", "entry_ts", "exit_ts", "entry_price",
    "exit_price", "stop_loss", "take_profit", "position_size", "risk_amount",
    "rr", "exit_reason", "pnl_pct", "pnl_usd", "r_multiple", "confidence",
    "regime_label", "duration_minutes", "mae_pct", "mfe_pct",
)

_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self'; "
        "base-uri 'none'; object-src 'none'; frame-ancestors 'none'")
VALIDATION_REFRESH_SECS = 900  # walk-forward is a slow-moving stat; no need
                               # to re-scan up to 100k trades more often

_SEC_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _dumps(o) -> str:
    return json.dumps(o, default=str)


def _clamp_limit(req, default: int, hi: int) -> int:
    """Parse ?limit safely: bad/negative values fall back, never 500."""
    try:
        v = int(req.query.get("limit", default))
    except (TypeError, ValueError):
        return default
    return max(1, min(v, hi))


async def _snapshot(db, starting_equity, info, learner_provider, heavy: bool) -> dict:
    """Build a live payload. Light fields every tick; heavy fields periodically."""
    payload: dict = {
        "open": await db.open_positions_count(),
        "equity": await db.latest_equity(starting_equity),
        "positions": [{k: r.get(k) for k in _TRADE_COLS}
                      for r in await db.get_open_trades()],
    }
    if heavy:
        m = await compute_metrics(db, starting_equity,
                                  universe=set(info.get("symbols") or []) or None)
        m["system"] = info
        payload["metrics"] = m
        closed = await db.get_closed_trades(limit=5000)
        payload["trades"] = [{k: r.get(k) for k in _TRADE_COLS}
                             for r in closed[-500:][::-1]]
        payload["analyses"] = await db.recent_analyses(250)
        payload["equity_curve"] = await db.equity_curve()
        payload["learner"] = learner_provider() if learner_provider else {}
    return payload


@web.middleware
async def _security_mw(request, handler):
    # --- optional Basic Auth (only enforced when DASHBOARD_PASS is set) ---
    want_pass = os.getenv("DASHBOARD_PASS", "")
    if want_pass:
        want_user = os.getenv("DASHBOARD_USER", "admin")
        ok = False
        hdr = request.headers.get("Authorization", "")
        if hdr.startswith("Basic "):
            try:
                user, _, pw = base64.b64decode(hdr[6:]).decode().partition(":")
                # constant-time compare to avoid timing oracles
                ok = (hmac.compare_digest(user, want_user)
                      and hmac.compare_digest(pw, want_pass))
            except Exception:
                ok = False
        if not ok:
            return web.Response(
                status=401, text="auth required",
                headers={"WWW-Authenticate": 'Basic realm="adaptive-trader"',
                         **_SEC_HEADERS})
    try:
        resp = await handler(request)
    except web.HTTPException as e:
        for k, v in _SEC_HEADERS.items():
            e.headers.setdefault(k, v)
        raise
    # A streaming response (SSE) is already prepared/sent — headers are frozen.
    if not getattr(resp, "prepared", False):
        for k, v in _SEC_HEADERS.items():
            resp.headers.setdefault(k, v)
    return resp


def build_app(db: Database, starting_equity: float,
              info: dict | None = None, learner_provider=None,
              cfg=None, feed=None, cycle_count_provider=None) -> web.Application:
    app = web.Application(middlewares=[_security_mw])
    info = info or {}

    async def index(_req):
        return web.Response(text=INDEX_PATH.read_text(encoding="utf-8"),
                            content_type="text/html")

    async def metrics(_req):
        m = await compute_metrics(db, starting_equity,
                                  universe=set(info.get("symbols") or []) or None)
        m["system"] = info
        return web.json_response(m, dumps=_dumps)

    async def positions(_req):
        rows = await db.get_open_trades()
        out = [{k: r.get(k) for k in _TRADE_COLS} for r in rows]
        return web.json_response(out, dumps=_dumps)

    async def trades(req):
        limit = _clamp_limit(req, 100, 1000)
        rows = await db.get_closed_trades(limit=5000)
        rows = rows[-limit:][::-1]  # newest first
        out = [{k: r.get(k) for k in _TRADE_COLS} for r in rows]
        return web.json_response(out, dumps=_dumps)

    async def analyses(req):
        limit = _clamp_limit(req, 40, 500)
        return web.json_response(await db.recent_analyses(limit), dumps=_dumps)

    async def equity(_req):
        return web.json_response(await db.equity_curve(), dumps=_dumps)

    async def live(req):
        """Cheap per-second payload (no metrics scan): equity, open, positions,
        + a shadow-labeling heartbeat so the training signal is visibly moving.
        Polled every 1s by the client for lag-free liveness through any proxy.
        ?ev_after=<id> returns only activity-feed events newer than that id."""
        payload = {
            "open": await db.open_positions_count(),
            "equity": await db.latest_equity(starting_equity),
            "positions": [{k: r.get(k) for k in _TRADE_COLS}
                          for r in await db.get_open_trades()],
        }
        try:
            payload["shadow"] = await db.shadow_counts()
        except Exception:
            pass
        try:
            engine_count = cycle_count_provider() if cycle_count_provider else 0
            payload["cycle_count"] = engine_count + validation_cache["refresh_count"]
        except Exception:
            pass
        if feed is not None:
            try:
                after = int(req.query.get("ev_after", 0))
            except (TypeError, ValueError):
                after = 0
            payload["events"] = feed.tail(max(0, after))
            payload["events_top"] = feed.top
        return web.json_response(payload, dumps=_dumps)

    async def learner(_req):
        """Extractable learning state: model weights + calibration + shadow."""
        snap = learner_provider() if learner_provider else {}
        try:
            snap["shadow"] = await db.shadow_counts()
        except Exception:
            pass
        return web.json_response(snap, dumps=_dumps)

    validation_cache: dict = {"result": None, "computed_at": 0.0, "refresh_count": 0}

    async def _recompute_validation():
        trades = await db.get_closed_trades(limit=100000)
        analyses = await db.recent_analyses(5000)
        result = run_walk_forward(trades, analyses, cfg)
        result["computed_at"] = time.time()
        result["refresh_interval_secs"] = VALIDATION_REFRESH_SECS
        validation_cache["result"] = result
        validation_cache["computed_at"] = result["computed_at"]
        validation_cache["refresh_count"] += 1
        return result

    async def _validation_refresh_loop():
        while True:
            try:
                await _recompute_validation()
            except Exception as e:
                log.warning("validation_refresh_failed", error=str(e))
            await asyncio.sleep(VALIDATION_REFRESH_SECS)

    async def validation(req):
        """Walk-forward / prequential out-of-sample edge validation.

        Recomputed on a background timer (every VALIDATION_REFRESH_SECS)
        rather than per-request, since it's a heavy scan over closed trades
        and the underlying stat only moves as fast as new trades close.
        Pass ?refresh=1 to force an immediate recompute.
        """
        if req.query.get("refresh") == "1" or validation_cache["result"] is None:
            result = await _recompute_validation()
        else:
            result = validation_cache["result"]
        return web.json_response(result, dumps=_dumps)

    app["_validation_refresh_task"] = None

    async def _on_startup(_app):
        _app["_validation_refresh_task"] = asyncio.create_task(
            _validation_refresh_loop())

    async def _on_cleanup(_app):
        task = _app.get("_validation_refresh_task")
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    async def stream(request):
        """Server-Sent Events: push live state every second (server-driven,
        no client polling). Light fields each tick; heavy fields every 5s."""
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable proxy buffering (Railway edge)
            "Connection": "keep-alive",
        })
        await resp.prepare(request)
        seq = 0
        try:
            while True:
                snap = await _snapshot(db, starting_equity, info,
                                       learner_provider, heavy=(seq % 5 == 0))
                snap["seq"] = seq
                await resp.write(b"data: " + _dumps(snap).encode() + b"\n\n")
                seq += 1
                await asyncio.sleep(1)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception as e:  # client dropped / write failed — end cleanly
            log.info("sse_closed", error=str(e))
        return resp

    async def trades_csv(_req):
        """Full closed-trade history as CSV for offline analysis."""
        rows = await db.get_closed_trades(limit=100000)
        buf = io.StringIO()
        if rows:
            cols = list(rows[0].keys())
            w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return web.Response(
            text=buf.getvalue(), content_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="trades.csv"'})

    app.router.add_get("/", index)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/positions", positions)
    app.router.add_get("/trades", trades)
    app.router.add_get("/analyses", analyses)
    app.router.add_get("/equity", equity)
    app.router.add_get("/live", live)
    app.router.add_get("/learner", learner)
    app.router.add_get("/validation", validation)
    app.router.add_get("/stream", stream)
    app.router.add_get("/export/trades.csv", trades_csv)
    return app


async def start_dashboard(db: Database, starting_equity: float,
                          host: str, port: int,
                          info: dict | None = None,
                          learner_provider=None, cfg=None,
                          feed=None, cycle_count_provider=None) -> web.AppRunner:
    if not os.getenv("DASHBOARD_PASS"):
        log.warning("dashboard_open",
                    hint="no DASHBOARD_PASS set — dashboard is publicly "
                         "readable. Set DASHBOARD_PASS to require Basic Auth.")
    app = build_app(db, starting_equity, info, learner_provider, cfg, feed,
                    cycle_count_provider)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner
