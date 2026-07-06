"""Execution-layer safety gates:
  * dry-run never touches the network
  * live requires enabled + keys + LIVE_CONFIRM=YES (explicit arming interlock)
  * live sizes are clamped to the notional cap
  * non-finite price/size is refused, never sent
"""
import pytest

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


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setenv("LIVE_CONFIRM", "YES")


@pytest.fixture
def disarmed(monkeypatch):
    monkeypatch.delenv("LIVE_CONFIRM", raising=False)


class FakeExchange:
    def __init__(self):
        self.orders = []
        self.sandbox = False

    def set_sandbox_mode(self, on):
        self.sandbox = on

    def amount_to_precision(self, _s, q):
        return f"{float(q):.8f}"

    def price_to_precision(self, _s, p):
        return f"{float(p):.2f}"

    async def create_order(self, symbol, typ, side, qty, price, params):
        self.orders.append({"symbol": symbol, "type": typ, "side": side,
                            "qty": qty, "params": params})
        return {"id": f"fake-{len(self.orders)}"}

    async def close(self):
        pass


async def test_disabled_executor_never_creates_client(armed):
    ex = BybitExecutor(_cfg(live=False, key="k", secret="s"))
    assert ex.live is False
    assert await ex.open_position(TRADE) is None
    assert await ex.close_position(TRADE) is None
    assert ex._exchange is None  # no ccxt client was ever constructed
    await ex.close()


async def test_enabled_without_keys_stays_dry_run(armed):
    ex = BybitExecutor(_cfg(live=True))  # no keys
    assert ex.live is False
    assert await ex.open_position(TRADE) is None
    assert ex._exchange is None
    await ex.close()


async def test_enabled_with_keys_but_not_confirmed_stays_dry_run(disarmed):
    # The arming interlock: keys present, live.enabled true, but LIVE_CONFIRM unset.
    ex = BybitExecutor(_cfg(live=True, key="k", secret="s"))
    assert ex.live is False
    assert await ex.open_position(TRADE) is None
    assert ex._exchange is None
    await ex.close()


async def test_live_clamps_qty_to_notional_cap(armed):
    ex = BybitExecutor(_cfg(live=True, key="k", secret="s"))
    fake = FakeExchange()
    ex._exchange = fake  # bypass real client construction
    assert ex.live is True

    big = dict(TRADE, position_size=1.0)  # 50k notional >> 200 cap
    oid = await ex.open_position(big)
    assert oid == "fake-1"
    o = fake.orders[0]
    assert o["side"] == "buy" and o["type"] == "market"
    assert abs(o["qty"] - 200.0 / 50000.0) < 1e-9  # clamped to cap
    assert o["params"]["stopLoss"] == "49000.00"
    assert o["params"]["takeProfit"] == "52000.00"

    await ex.close_position(big)
    c = fake.orders[1]
    assert c["side"] == "sell" and c["params"]["reduceOnly"] is True
    assert abs(c["qty"] - 200.0 / 50000.0) < 1e-9  # close mirrors clamped qty


async def test_live_small_order_not_clamped(armed):
    ex = BybitExecutor(_cfg(live=True, key="k", secret="s"))
    fake = FakeExchange()
    ex._exchange = fake
    oid = await ex.open_position(TRADE)  # 0.002 * 50k = $100 < cap
    assert oid == "fake-1"
    assert abs(fake.orders[0]["qty"] - 0.002) < 1e-9


@pytest.mark.parametrize("bad", [0.0, -50000.0, float("nan"), float("inf")])
async def test_live_refuses_non_finite_price(armed, bad):
    ex = BybitExecutor(_cfg(live=True, key="k", secret="s"))
    fake = FakeExchange()
    ex._exchange = fake
    assert ex.live is True
    trade = dict(TRADE, entry_price=bad)
    assert await ex.open_position(trade) is None
    assert fake.orders == []  # nothing sent — cap bypass closed
