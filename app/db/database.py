"""SQLite persistence (async). Schema is plain SQL, easy to migrate to PostgreSQL."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    bias TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw_prob REAL,
    regime_label TEXT,
    regime_posterior TEXT,
    changepoint_prob REAL,
    features TEXT NOT NULL,
    trade_recommended INTEGER NOT NULL,
    reasoning TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,             -- long / short
    status TEXT NOT NULL DEFAULT 'open', -- open / closed
    entry_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    position_size REAL NOT NULL,         -- base asset units
    risk_amount REAL NOT NULL,           -- $ at risk if SL hit
    rr REAL NOT NULL,
    confidence REAL NOT NULL,
    raw_prob REAL,
    regime_label TEXT,
    changepoint_prob REAL,
    features TEXT NOT NULL,              -- entry FeatureVector json
    reasoning TEXT,
    similar_trades TEXT,                 -- retrieval summary json at entry
    -- exit fields
    exit_ts TEXT,
    exit_price REAL,
    exit_reason TEXT,                    -- tp / sl / regime_exit / changepoint_exit / time_exit
    pnl_usd REAL,
    pnl_pct REAL,
    r_multiple REAL,
    duration_minutes REAL,
    mae_pct REAL,                        -- max adverse excursion while open
    mfe_pct REAL,                        -- max favorable excursion while open
    exit_features TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL REFERENCES trades(id),
    ts TEXT NOT NULL,
    review TEXT NOT NULL,                -- json: structured post-trade review
    lessons TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    event TEXT
);
-- live order tracking (only populated when real Bybit orders are sent) so a
-- restart can reconcile persisted state against the exchange's positions.
CREATE TABLE IF NOT EXISTS live_orders (
    order_id TEXT PRIMARY KEY,           -- our id: "t<trade_id>"
    trade_id INTEGER,                    -- paper trade this mirrors
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    bybit_order_id TEXT,                 -- exchange order id from create_order
    order_status TEXT NOT NULL,          -- pending / open / closed / unknown_state
    regime_state_at_entry TEXT,          -- json snapshot
    prediction_state_at_entry TEXT,      -- json: raw_prob, confidence, ...
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS order_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    event_type TEXT NOT NULL,            -- created / entry_filled / closed / reconcile_ok / reconcile_mismatch / app_shutdown
    details TEXT,                        -- json
    ts TEXT NOT NULL
);
-- shadow setups: the model's preferred direction + barrier every analysis,
-- resolved later against the forward price path (TP-before-SL) to generate
-- training labels far beyond the rare actual fills (fixes data starvation).
CREATE TABLE IF NOT EXISTS shadow_setups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_ts TEXT NOT NULL,
    entry_ms INTEGER NOT NULL,           -- entry candle unix ms (for resolution)
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    features TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0, -- 0 open, 1 resolved, 2 expired
    outcome INTEGER                      -- 1 tp-first, 0 sl-first
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_analyses_symbol_ts ON analyses(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_live_orders_status ON live_orders(order_status);
CREATE INDEX IF NOT EXISTS idx_order_events_order ON order_events(order_id);
CREATE INDEX IF NOT EXISTS idx_shadow_open ON shadow_setups(symbol, resolved);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str):
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected")
        return self._db

    # ---------- analyses ----------
    async def insert_analysis(self, a: dict[str, Any]) -> int:
        cur = await self.db.execute(
            """INSERT INTO analyses (ts, symbol, price, bias, confidence, raw_prob,
               regime_label, regime_posterior, changepoint_prob, features,
               trade_recommended, reasoning)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (a.get("ts", utcnow()), a["symbol"], a["price"], a["bias"],
             a["confidence"], a.get("raw_prob"), a.get("regime_label"),
             json.dumps(a.get("regime_posterior")), a.get("changepoint_prob"),
             json.dumps(a["features"]), int(a["trade_recommended"]),
             a.get("reasoning")),
        )
        await self.db.commit()
        return cur.lastrowid

    async def recent_analyses(self, limit: int = 40) -> list[dict]:
        cur = await self.db.execute(
            """SELECT id, ts, symbol, price, bias, confidence, raw_prob,
               regime_label, changepoint_prob, trade_recommended, reasoning
               FROM analyses ORDER BY id DESC LIMIT ?""", (limit,))
        return [dict(r) for r in await cur.fetchall()]

    # ---------- trades ----------
    async def open_trade(self, t: dict[str, Any]) -> int:
        cur = await self.db.execute(
            """INSERT INTO trades (symbol, direction, entry_ts, entry_price, stop_loss,
               take_profit, position_size, risk_amount, rr, confidence, raw_prob,
               regime_label, changepoint_prob, features, reasoning, similar_trades,
               mae_pct, mfe_pct)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)""",
            (t["symbol"], t["direction"], t.get("entry_ts", utcnow()),
             t["entry_price"], t["stop_loss"], t["take_profit"],
             t["position_size"], t["risk_amount"], t["rr"], t["confidence"],
             t.get("raw_prob"), t.get("regime_label"), t.get("changepoint_prob"),
             json.dumps(t["features"]), t.get("reasoning"),
             json.dumps(t.get("similar_trades"))),
        )
        await self.db.commit()
        return cur.lastrowid

    async def update_excursions(self, trade_id: int, mae_pct: float, mfe_pct: float) -> None:
        await self.db.execute(
            "UPDATE trades SET mae_pct=?, mfe_pct=? WHERE id=?",
            (mae_pct, mfe_pct, trade_id),
        )
        await self.db.commit()

    async def close_trade(self, trade_id: int, c: dict[str, Any]) -> None:
        await self.db.execute(
            """UPDATE trades SET status='closed', exit_ts=?, exit_price=?, exit_reason=?,
               pnl_usd=?, pnl_pct=?, r_multiple=?, duration_minutes=?, exit_features=?
               WHERE id=?""",
            (c.get("exit_ts", utcnow()), c["exit_price"], c["exit_reason"],
             c["pnl_usd"], c["pnl_pct"], c["r_multiple"], c["duration_minutes"],
             json.dumps(c.get("exit_features")), trade_id),
        )
        await self.db.commit()

    async def get_open_trades(self) -> list[dict]:
        cur = await self.db.execute("SELECT * FROM trades WHERE status='open'")
        return [dict(r) for r in await cur.fetchall()]

    async def get_trade(self, trade_id: int) -> dict | None:
        cur = await self.db.execute("SELECT * FROM trades WHERE id=?", (trade_id,))
        r = await cur.fetchone()
        return dict(r) if r else None

    async def get_closed_trades(self, symbol: str | None = None, limit: int = 5000) -> list[dict]:
        q = "SELECT * FROM trades WHERE status='closed'"
        args: list[Any] = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        q += " ORDER BY exit_ts ASC LIMIT ?"
        args.append(limit)
        cur = await self.db.execute(q, args)
        return [dict(r) for r in await cur.fetchall()]

    async def count_closed_trades(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) c FROM trades WHERE status='closed'")
        row = await cur.fetchone()
        return int(row["c"])

    async def open_positions_count(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) c FROM trades WHERE status='open'")
        row = await cur.fetchone()
        return int(row["c"])

    async def has_open_trade(self, symbol: str) -> bool:
        cur = await self.db.execute(
            "SELECT COUNT(*) c FROM trades WHERE status='open' AND symbol=?", (symbol,)
        )
        row = await cur.fetchone()
        return int(row["c"]) > 0

    async def last_exit_ts(self, symbol: str) -> str | None:
        cur = await self.db.execute(
            "SELECT exit_ts FROM trades WHERE status='closed' AND symbol=? "
            "ORDER BY exit_ts DESC LIMIT 1", (symbol,),
        )
        row = await cur.fetchone()
        return row["exit_ts"] if row else None

    # ---------- reviews ----------
    async def insert_review(self, trade_id: int, review: dict, lessons: str) -> int:
        cur = await self.db.execute(
            "INSERT INTO reviews (trade_id, ts, review, lessons) VALUES (?,?,?,?)",
            (trade_id, utcnow(), json.dumps(review), lessons),
        )
        await self.db.commit()
        return cur.lastrowid

    async def recent_lessons(self, symbol: str | None = None, limit: int = 10) -> list[str]:
        q = ("SELECT r.lessons FROM reviews r JOIN trades t ON t.id = r.trade_id ")
        args: list[Any] = []
        if symbol:
            q += "WHERE t.symbol=? "
            args.append(symbol)
        q += "ORDER BY r.id DESC LIMIT ?"
        args.append(limit)
        cur = await self.db.execute(q, args)
        return [r["lessons"] for r in await cur.fetchall() if r["lessons"]]

    # ---------- equity ----------
    async def record_equity(self, equity: float, event: str = "") -> None:
        await self.db.execute(
            "INSERT INTO equity (ts, equity, event) VALUES (?,?,?)",
            (utcnow(), equity, event),
        )
        await self.db.commit()

    async def latest_equity(self, default: float) -> float:
        cur = await self.db.execute("SELECT equity FROM equity ORDER BY id DESC LIMIT 1")
        row = await cur.fetchone()
        return float(row["equity"]) if row else default

    async def equity_curve(self) -> list[dict]:
        cur = await self.db.execute("SELECT ts, equity, event FROM equity ORDER BY id ASC")
        return [dict(r) for r in await cur.fetchall()]

    async def peak_equity(self, default: float) -> float:
        cur = await self.db.execute("SELECT MAX(equity) m FROM equity")
        row = await cur.fetchone()
        return float(row["m"]) if row and row["m"] is not None else default

    async def realized_pnl_today(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        cur = await self.db.execute(
            "SELECT COALESCE(SUM(pnl_usd),0) s FROM trades "
            "WHERE status='closed' AND substr(exit_ts,1,10)=?", (today,),
        )
        row = await cur.fetchone()
        return float(row["s"])

    # ---------- live order tracking (reconciliation) ----------
    async def record_live_order(self, o: dict[str, Any]) -> None:
        await self.db.execute(
            """INSERT OR REPLACE INTO live_orders
               (order_id, trade_id, symbol, direction, entry_price, quantity,
                bybit_order_id, order_status, regime_state_at_entry,
                prediction_state_at_entry, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (o["order_id"], o.get("trade_id"), o["symbol"], o["direction"],
             o["entry_price"], o["quantity"], o.get("bybit_order_id"),
             o.get("order_status", "pending"),
             json.dumps(o.get("regime_state_at_entry")),
             json.dumps(o.get("prediction_state_at_entry")),
             o.get("created_at", utcnow()), utcnow()),
        )
        await self.db.commit()

    async def set_live_order(self, order_id: str, *, status: str | None = None,
                             bybit_order_id: str | None = None) -> None:
        sets, args = ["updated_at=?"], [utcnow()]
        if status is not None:
            sets.append("order_status=?"); args.append(status)
        if bybit_order_id is not None:
            sets.append("bybit_order_id=?"); args.append(bybit_order_id)
        args.append(order_id)
        await self.db.execute(
            f"UPDATE live_orders SET {', '.join(sets)} WHERE order_id=?", args)
        await self.db.commit()

    async def open_live_orders(self) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM live_orders WHERE order_status IN "
            "('pending','open','unknown_state')")
        return [dict(r) for r in await cur.fetchall()]

    async def add_order_event(self, order_id: str, event_type: str,
                              details: dict | None = None) -> None:
        await self.db.execute(
            "INSERT INTO order_events (order_id, event_type, details, ts) "
            "VALUES (?,?,?,?)",
            (order_id, event_type, json.dumps(details or {}), utcnow()))
        await self.db.commit()

    # ---------- shadow setups (training-label generation) ----------
    async def record_shadow_setup(self, s: dict[str, Any]) -> None:
        await self.db.execute(
            """INSERT INTO shadow_setups
               (symbol, direction, entry_ts, entry_ms, entry_price, stop_loss,
                take_profit, features, resolved)
               VALUES (?,?,?,?,?,?,?,?,0)""",
            (s["symbol"], s["direction"], s.get("entry_ts", utcnow()),
             int(s["entry_ms"]), s["entry_price"], s["stop_loss"],
             s["take_profit"], json.dumps(s["features"])),
        )
        await self.db.commit()

    async def open_shadow_setups(self, symbol: str, limit: int = 500) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM shadow_setups WHERE symbol=? AND resolved=0 "
            "ORDER BY id ASC LIMIT ?", (symbol, limit))
        return [dict(r) for r in await cur.fetchall()]

    async def resolve_shadow_setup(self, sid: int, resolved: int,
                                   outcome: int | None) -> None:
        await self.db.execute(
            "UPDATE shadow_setups SET resolved=?, outcome=? WHERE id=?",
            (resolved, outcome, sid))
        await self.db.commit()

    async def shadow_counts(self) -> dict:
        cur = await self.db.execute(
            "SELECT resolved, COUNT(*) c, COALESCE(SUM(outcome),0) wins "
            "FROM shadow_setups GROUP BY resolved")
        rows = {r["resolved"]: (r["c"], r["wins"]) for r in await cur.fetchall()}
        resolved = rows.get(1, (0, 0))
        return {"open": rows.get(0, (0, 0))[0], "resolved": resolved[0],
                "expired": rows.get(2, (0, 0))[0],
                "resolved_win_rate": (round(resolved[1] / resolved[0], 4)
                                      if resolved[0] else None)}
