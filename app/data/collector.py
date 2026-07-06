"""Read-only market data collection via ccxt (async).

This module never creates orders and passes no API keys — only public
endpoints (OHLCV, ticker, order book). Order execution lives in a separate,
gated layer (app/trading/executor.py); this collector stays read-only.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import ccxt.async_support as ccxt
import numpy as np

from app.logging_setup import get_logger

log = get_logger("collector")


@dataclass
class MarketSnapshot:
    symbol: str
    timeframe: str
    ts: np.ndarray        # unix ms, shape (n,)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    last_price: float
    bid: float | None = None
    ask: float | None = None
    ob_imbalance: float | None = None   # (bid_vol - ask_vol) / (bid_vol + ask_vol), top 10 levels

    @property
    def n(self) -> int:
        return len(self.close)


class MarketDataCollector:
    def __init__(self, exchange_id: str, timeframe: str, lookback: int,
                 max_retries: int = 4, market_type: str = "swap"):
        klass = getattr(ccxt, exchange_id, None)
        if klass is None:
            raise ValueError(f"Unknown ccxt exchange id: {exchange_id}")
        # No apiKey / secret: public data only. This client cannot place orders;
        # order execution lives in app/trading/executor.py behind live.enabled.
        self.exchange = klass({
            "enableRateLimit": True,
            "options": {"defaultType": market_type},
        })
        self.timeframe = timeframe
        self.lookback = lookback
        self.max_retries = max_retries

    async def close(self) -> None:
        await self.exchange.close()

    async def _with_retries(self, coro_factory, what: str):
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                return await coro_factory()
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, asyncio.TimeoutError) as e:
                log.warning("fetch_retry", what=what, attempt=attempt, error=str(e))
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def fetch_snapshot(self, symbol: str) -> MarketSnapshot:
        ohlcv = await self._with_retries(
            lambda: self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.lookback),
            f"ohlcv:{symbol}",
        )
        arr = np.asarray(ohlcv, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 10:
            raise RuntimeError(f"Insufficient OHLCV for {symbol}: {arr.shape}")

        bid = ask = imb = None
        try:
            ob = await self._with_retries(
                lambda: self.exchange.fetch_order_book(symbol, limit=10),
                f"orderbook:{symbol}",
            )
            bids, asks = ob.get("bids") or [], ob.get("asks") or []
            if bids and asks:
                bid, ask = float(bids[0][0]), float(asks[0][0])
                bv = sum(float(b[1]) for b in bids[:10])
                av = sum(float(a[1]) for a in asks[:10])
                if bv + av > 0:
                    imb = (bv - av) / (bv + av)
        except Exception as e:  # order book is a nice-to-have, never fatal
            log.warning("orderbook_unavailable", symbol=symbol, error=str(e))

        return MarketSnapshot(
            symbol=symbol,
            timeframe=self.timeframe,
            ts=arr[:, 0],
            open=arr[:, 1],
            high=arr[:, 2],
            low=arr[:, 3],
            close=arr[:, 4],
            volume=arr[:, 5],
            last_price=float(arr[-1, 4]),
            bid=bid,
            ask=ask,
            ob_imbalance=imb,
        )

    async def fetch_last_price(self, symbol: str) -> float:
        t = await self._with_retries(
            lambda: self.exchange.fetch_ticker(symbol), f"ticker:{symbol}"
        )
        price = t.get("last") or t.get("close")
        if price is None:
            raise RuntimeError(f"No last price for {symbol}")
        return float(price)
