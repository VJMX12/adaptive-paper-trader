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

import math
import os

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
        # Explicit arming interlock: provisioning API keys is NOT enough to
        # place real orders. The operator must also deliberately set
        # LIVE_CONFIRM=YES. This prevents "added keys → suddenly live".
        self._confirmed = os.getenv("LIVE_CONFIRM", "").strip().upper() == "YES"
        self._testnet = bool(cfg.get("live.testnet", False))
        has_keys = bool(self._key and self._secret)
        self.live = enabled and has_keys and self._confirmed
        if enabled and has_keys and not self._confirmed:
            log.warning("live_not_confirmed",
                        hint="live.enabled=true and keys present, but "
                             "LIVE_CONFIRM!=YES — staying dry-run. Set env "
                             "LIVE_CONFIRM=YES to arm real orders.")
        elif enabled and not has_keys:
            log.warning("live_disabled_no_keys",
                        hint="live.enabled is true but BYBIT_API_KEY/"
                             "BYBIT_API_SECRET missing — staying dry-run")
        self._exchange = None
        log.info("executor_mode", live=self.live, market_type=self.market_type,
                 max_notional_usd=self.max_notional, confirmed=self._confirmed)

    async def _client(self):
        if self._exchange is None:
            ex = ccxt.bybit({
                "apiKey": self._key,
                "secret": self._secret,
                "enableRateLimit": True,
                "options": {"defaultType": self.market_type},
            })
            if self._testnet:
                ex.set_sandbox_mode(True)  # Bybit testnet — validate wiring, no real money
            self._exchange = ex
            await self._exchange.load_markets()
        return self._exchange

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    def live_qty(self, trade: dict) -> float | None:
        """Public: the actual (clamped) qty that would be sent for this trade."""
        return self._live_qty(trade)

    async def fetch_open_positions(self) -> dict:
        """symbol -> position dict for every non-zero live position on Bybit.
        Empty when not live (nothing on the exchange to reconcile against)."""
        if not self.live:
            return {}
        ex = await self._client()
        out: dict = {}
        for p in await ex.fetch_positions():
            contracts = p.get("contracts") or 0
            if contracts and abs(float(contracts)) > 0:
                out[p["symbol"]] = p
        return out

    # ------------------------------------------------------------------
    def _live_qty(self, trade: dict) -> float | None:
        """Paper position size, clamped so entry notional <= the cap.

        Returns None if the price or size is not a positive, finite number —
        a defensive refusal so a bad value can never bypass the cap and send
        an unclamped real order.
        """
        try:
            qty = float(trade["position_size"])
            entry = float(trade["entry_price"])
        except (TypeError, ValueError, KeyError):
            return None
        if not (math.isfinite(qty) and math.isfinite(entry)
                and qty > 0 and entry > 0):
            return None
        if qty * entry > self.max_notional:
            qty = self.max_notional / entry
        return qty

    def _intent(self, trade: dict, side: str, ref_price: float) -> dict | None:
        qty = self._live_qty(trade)
        if qty is None:
            return None
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
        intent = self._intent(trade, side, float(trade.get("entry_price") or 0))
        if intent is None:
            log.warning("live_order_refused", reason="non-finite price/size",
                        trade_id=trade.get("id"), symbol=trade.get("symbol"))
            return None
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
        ref = float(trade.get("exit_price") or trade.get("entry_price") or 0)
        intent = self._intent(trade, side, ref)
        if intent is None:
            log.warning("live_close_refused", reason="non-finite price/size",
                        trade_id=trade.get("id"), symbol=trade.get("symbol"))
            return None
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
