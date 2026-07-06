"""Dashboard server: aiohttp JSON API + single-page command deck UI.

Endpoints:
  /            HTML dashboard (app/dashboard/index.html)
  /metrics     performance metrics + system info (mode, exchange, symbols)
  /positions   open paper trades
  /trades      recent closed trades (newest first)
  /analyses    recent analyses (newest first)
  /equity      equity curve
"""
from __future__ import annotations

import json
from pathlib import Path

from aiohttp import web

from app.dashboard.metrics import compute_metrics
from app.db.database import Database

INDEX_PATH = Path(__file__).with_name("index.html")

_TRADE_COLS = (
    "id", "symbol", "direction", "entry_ts", "exit_ts", "entry_price",
    "exit_price", "stop_loss", "take_profit", "position_size", "risk_amount",
    "rr", "exit_reason", "pnl_pct", "pnl_usd", "r_multiple", "confidence",
    "regime_label", "duration_minutes", "mae_pct", "mfe_pct",
)


def _dumps(o) -> str:
    return json.dumps(o, default=str)


def build_app(db: Database, starting_equity: float,
              info: dict | None = None) -> web.Application:
    app = web.Application()
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
        limit = min(int(req.query.get("limit", 100)), 1000)
        rows = await db.get_closed_trades(limit=5000)
        rows = rows[-limit:][::-1]  # newest first
        out = [{k: r.get(k) for k in _TRADE_COLS} for r in rows]
        return web.json_response(out, dumps=_dumps)

    async def analyses(req):
        limit = min(int(req.query.get("limit", 40)), 500)
        return web.json_response(await db.recent_analyses(limit), dumps=_dumps)

    async def equity(_req):
        return web.json_response(await db.equity_curve(), dumps=_dumps)

    app.router.add_get("/", index)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/positions", positions)
    app.router.add_get("/trades", trades)
    app.router.add_get("/analyses", analyses)
    app.router.add_get("/equity", equity)
    return app


async def start_dashboard(db: Database, starting_equity: float,
                          host: str, port: int,
                          info: dict | None = None) -> web.AppRunner:
    app = build_app(db, starting_equity, info)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner
