"""Bybit order execution layer — DRY-RUN BY DEFAULT.

Mirrors the paper engine's decisions onto Bybit:
  * paper trade opened  -> market order  (+ exchange-side SL/TP on swap)
  * paper trade closed  -> reduce-only market order in the opposite side

Safety gates, in order:
  1. ``live.enabled`` in config must be true       (default: false)
  2. BYBIT_API_KEY / BYBIT_API_SECRET must be set in .env
  3. per-order notional cap (``live.max_notional_usd``): the paper size is
     CLAMPED down so entry notional never exceeds the cap — the paper brain
     can size for $10k of simulated equity while real orders stay small.

While gate 1 or 2 fails, every order intent is logged in full
(``live_dry_run_*`` events) so the numbers can be studied — but nothing
is ever transmitted to the exchange.
"""
from __future__ import annotations

import ccxt.async_support as ccxt

from app.logging_setup import get_logger

log = get_logger("executor")


class BybitExecutor:
    def __init__(self, cfg):
        self.market_type = str(cfg.get("exchange.market_type", "swap"))
        self.max_notional = float(cfg.get("live.max_notional_usd", 200.0))
        self._key = cfg.secrets.bybit_api_key or ""
        self._secret = cfg.secrets.bybit_api_secret or ""
        enabled = bool(cfg.get("live.enabled", False))
        self.live = enabled and bool(self._key and self._secret)
        if enabled and not self.live:
            log.warning("live_disabled_no_keys",
                        hint="live.enabled is true but BYBIT_API_KEY/"
                             "BYBIT_API_SECRET missing in .env — staying dry-run")
        self._exchange = None
        log.info("executor_mode", live=self.live, market_type=self.market_type,
                 max_notional_usd=self.max_notional)

    async def _client(self):
        if self._exchange is None:
            self._exchange = ccxt.bybit({
                "apiKey": self._key,
                "secret": self._secret,
                "enableRateLimit": True,
                "options": {"defaultType": self.market_type},
            })
            await self._exchange.load_markets()
        return self._exchange

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    # ------------------------------------------------------------------
    def _live_qty(self, trade: dict) -> float:
        """Paper position size, clamped so entry notional <= the cap."""
        qty = float(trade["position_size"])
        entry = float(trade["entry_price"])
        if entry > 0 and qty * entry > self.max_notional:
            qty = self.max_notional / entry
        return qty

    def _intent(self, trade: dict, side: str, ref_price: float) -> dict:
        qty = self._live_qty(trade)
        return {
            "symbol": trade["symbol"], "side": side, "type": "market",
            "qty": round(qty, 8), "ref_price": ref_price,
            "notional_usd": round(qty * ref_price, 2),
            "paper_qty": round(float(trade["position_size"]), 8),
            "market_type": self.market_type,
        }

    async def open_position(self, trade: dict) -> str | None:
        """Mirror a freshly opened paper trade as a real Bybit order."""
        side = "buy" if trade["direction"] == "long" else "sell"
        intent = self._intent(trade, side, float(trade["entry_price"]))
        intent["stop_loss"] = trade.get("stop_loss")
        intent["take_profit"] = trade.get("take_profit")

        if not self.live:
            log.info("live_dry_run_open", sent=False,
                     note="order NOT sent (dry-run)", **intent)
            return None
        if intent["qty"] < intent["paper_qty"]:
            log.info("live_qty_clamped", cap_usd=self.max_notional, **intent)

        ex = await self._client()
        symbol = trade["symbol"]
        qty = float(ex.amount_to_precision(symbol, intent["qty"]))
        params: dict = {}
        if self.market_type == "swap":
            params["positionIdx"] = 0  # one-way mode
            if trade.get("stop_loss"):
                params["stopLoss"] = ex.price_to_precision(symbol, trade["stop_loss"])
            if trade.get("take_profit"):
                params["takeProfit"] = ex.price_to_precision(symbol, trade["take_profit"])
        order = await ex.create_order(symbol, "market", side, qty, None, params)
        log.info("live_order_sent", order_id=order.get("id"), **intent)
        return order.get("id")

    async def close_position(self, trade: dict) -> str | None:
        """Mirror a paper-trade close as a reduce-only market order."""
        side = "sell" if trade["direction"] == "long" else "buy"
        ref = float(trade.get("exit_price") or trade["entry_price"])
        intent = self._intent(trade, side, ref)
        intent["exit_reason"] = trade.get("exit_reason")

        if not self.live:
            log.info("live_dry_run_close", sent=False,
                     note="order NOT sent (dry-run)", **intent)
            return None

        ex = await self._client()
        symbol = trade["symbol"]
        qty = float(ex.amount_to_precision(symbol, intent["qty"]))
        params: dict = {}
        if self.market_type == "swap":
            params = {"reduceOnly": True, "positionIdx": 0}
        order = await ex.create_order(symbol, "market", side, qty, None, params)
        log.info("live_order_sent", order_id=order.get("id"), **intent)
        return order.get("id")
