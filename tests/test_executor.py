"""Execution-layer safety gates: dry-run must never touch the network."""
from app.config import AppConfig, Secrets
from app.trading.executor import BybitExecutor

TRADE = {
    "symbol": "BTC/USDT:USDT", "direction": "long",
    "entry_price": 50000.0, "position_size": 0.002,
    "stop_loss": 49000.0, "take_profit": 52000.0,
    "exit_price": 51000.0, "exit_reason": "tp",
}


def _cfg(live: bool, key: str = "", secret: str = "") -> AppConfig:
    return AppConfig(
        raw={"exchange": {"market_type": "swap"},
             "live": {"enabled": live, "max_notional_usd": 200.0}},
        secrets=Secrets(bybit_api_key=key, bybit_api_secret=secret),
    )


async def test_disabled_executor_never_creates_client():
    ex = BybitExecutor(_cfg(live=False, key="k", secret="s"))
    assert ex.live is False
    assert await ex.open_position(TRADE) is None
    assert await ex.close_position(TRADE) is None
    assert ex._exchange is None  # no ccxt client was ever constructed
    await ex.close()


async def test_enabled_without_keys_stays_dry_run():
    ex = BybitExecutor(_cfg(live=True))  # no keys in .env
    assert ex.live is False
    assert await ex.open_position(TRADE) is None
    assert ex._exchange is None
    await ex.close()


async def test_live_blocks_orders_over_notional_cap():
    ex = BybitExecutor(_cfg(live=True, key="k", secret="s"))
    assert ex.live is True
    big = dict(TRADE, position_size=1.0)  # 50k notional >> 200 cap
    assert await ex.open_position(big) is None
    assert ex._exchange is None  # blocked before any client/network use
    await ex.close()
