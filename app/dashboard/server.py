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

import base64
import csv
import hmac
import io
import json
import os
from pathlib import Path

from aiohttp import web

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
    for k, v in _SEC_HEADERS.items():
        resp.headers.setdefault(k, v)
    return resp


def build_app(db: Database, starting_equity: float,
              info: dict | None = None, learner_provider=None) -> web.Application:
    app = web.Application(middlewares=[_security_mw])
    info = info or {}

    async def index(_req):
        return web.Response(text=INDEX_PATH.read_text(encoding="utf-8"),
                            content_type="text/html")

    async def metrics(_req):
        m = await compute_metrics(db, starting_equity)
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

    async def learner(_req):
        """Extractable learning state: model weights + calibration quality."""
        snap = learner_provider() if learner_provider else {}
        return web.json_response(snap, dumps=_dumps)

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
    app.router.add_get("/learner", learner)
    app.router.add_get("/export/trades.csv", trades_csv)
    return app


async def start_dashboard(db: Database, starting_equity: float,
                          host: str, port: int,
                          info: dict | None = None,
                          learner_provider=None) -> web.AppRunner:
    if not os.getenv("DASHBOARD_PASS"):
        log.warning("dashboard_open",
                    hint="no DASHBOARD_PASS set — dashboard is publicly "
                         "readable. Set DASHBOARD_PASS to require Basic Auth.")
    app = build_app(db, starting_equity, info, learner_provider)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner
