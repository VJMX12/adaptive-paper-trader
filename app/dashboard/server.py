"""Minimal performance dashboard: aiohttp server with /metrics JSON and an HTML view."""
from __future__ import annotations

import json

from aiohttp import web

from app.dashboard.metrics import compute_metrics
from app.db.database import Database

HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Adaptive Paper Trader</title>
<meta http-equiv="refresh" content="60">
<style>
body{font-family:ui-monospace,Menlo,monospace;background:#0b1a15;color:#a5e983;
padding:24px;max-width:960px;margin:auto}
h1{color:#e8fff0;font-size:20px} .card{background:#10241c;border:1px solid #1e3d30;
border-radius:10px;padding:16px;margin:12px 0} .k{color:#6fae8f}
table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid #1e3d30;
padding:6px 8px;text-align:left;font-size:13px} pre{white-space:pre-wrap}
.neg{color:#ff8a8a}.pos{color:#a5e983}
</style></head><body>
<h1>📊 Adaptive Paper Trader — live metrics (auto-refresh 60s)</h1>
<div class="card"><pre id="m">loading…</pre></div>
<script>
fetch('/metrics').then(r=>r.json()).then(d=>{
  document.getElementById('m').textContent = JSON.stringify(d,null,2);
});
</script></body></html>"""


def build_app(db: Database, starting_equity: float) -> web.Application:
    app = web.Application()

    async def metrics(_req):
        m = await compute_metrics(db, starting_equity)
        return web.json_response(m, dumps=lambda o: json.dumps(o, default=str))

    async def index(_req):
        return web.Response(text=HTML, content_type="text/html")

    async def trades(_req):
        rows = await db.get_closed_trades(limit=200)
        slim = [{k: r[k] for k in (
            "id", "symbol", "direction", "entry_ts", "exit_ts", "entry_price",
            "exit_price", "exit_reason", "pnl_pct", "pnl_usd", "r_multiple",
            "confidence", "regime_label")} for r in rows]
        return web.json_response(slim, dumps=lambda o: json.dumps(o, default=str))

    app.router.add_get("/", index)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/trades", trades)
    return app


async def start_dashboard(db: Database, starting_equity: float,
                          host: str, port: int) -> web.AppRunner:
    app = build_app(db, starting_equity)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner
