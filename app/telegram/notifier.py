"""Telegram notifications via raw Bot API (sendMessage), with retries.

If telegram.enabled is false or credentials are missing, every call is a
no-op — the system runs fine without it.
"""
from __future__ import annotations

import asyncio

import aiohttp

from app.analysis.engine import AnalysisResult
from app.logging_setup import get_logger

log = get_logger("telegram")


class TelegramNotifier:
    def __init__(self, cfg):
        self.enabled = bool(cfg.get("telegram.enabled"))
        self.token = cfg.secrets.telegram_bot_token
        self.chat_id = cfg.secrets.telegram_chat_id
        if self.enabled and not (self.token and self.chat_id):
            self.enabled = False

    async def send(self, text: str, retries: int = 3) -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                   "disable_web_page_preview": True}
        delay = 1.0
        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(url, json=payload,
                                      timeout=aiohttp.ClientTimeout(total=15)) as r:
                        if r.status == 200:
                            return True
                        body = await r.text()
                        log.warning("telegram_http", status=r.status, body=body[:200])
            except Exception as e:
                log.warning("telegram_error", attempt=attempt, error=str(e))
            await asyncio.sleep(delay)
            delay *= 2
        return False

    # ---------- formatted events ----------
    async def analysis_update(self, res: AnalysisResult, reasoning: str) -> None:
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}[res.bias]
        rec = "✅ trade recommended" if res.trade_recommended else \
            "🚫 no trade — " + "; ".join(res.veto_reasons)
        text = (
            f"{emoji} <b>{res.symbol}</b> analysis\n"
            f"Price: <code>{res.price:.6g}</code>\n"
            f"Bias: <b>{res.bias}</b> | Confidence: <b>{res.confidence:.0%}</b>\n"
            f"Regime: {res.regime_label}\n"
            f"Changepoint prob: {res.changepoint_prob:.2f}\n"
            f"{rec}\n\n<i>{_trim(reasoning, 500)}</i>"
        )
        await self.send(text)

    async def trade_opened(self, trade_id: int, res: AnalysisResult,
                           size: float, risk_amount: float, reasoning: str) -> None:
        lo, hi = res.entry_zone
        text = (
            f"📈 <b>PAPER TRADE OPENED #{trade_id}</b>\n"
            f"Pair: <b>{res.symbol}</b> | Direction: <b>{res.direction.upper()}</b>\n"
            f"Entry: <code>{res.price:.6g}</code> (zone {lo:.6g}–{hi:.6g})\n"
            f"SL: <code>{res.stop_loss:.6g}</code> | TP: <code>{res.take_profit:.6g}</code>\n"
            f"R/R: <b>{res.rr}</b> | Confidence: <b>{res.confidence:.0%}</b>\n"
            f"Size: {size:.6g} | Risk: ${risk_amount:.2f}\n"
            f"Regime: {res.regime_label}\n\n<i>{_trim(reasoning, 600)}</i>"
        )
        await self.send(text)

    async def trade_closed(self, trade: dict, review_lessons: str,
                           equity: float) -> None:
        won = (trade.get("pnl_usd") or 0) > 0
        emoji = "✅" if won else "❌"
        dur_h = (trade.get("duration_minutes") or 0) / 60
        text = (
            f"{emoji} <b>PAPER TRADE CLOSED #{trade['id']}</b>\n"
            f"Pair: <b>{trade['symbol']}</b> ({trade['direction']})\n"
            f"P/L: <b>{trade['pnl_pct']:+.2f}%</b> "
            f"(<b>${trade['pnl_usd']:+.2f}</b>, {trade['r_multiple']:+.2f}R)\n"
            f"Duration: {dur_h:.1f}h | Exit: <b>{trade['exit_reason']}</b>\n"
            f"Entry {float(trade['entry_price']):.6g} → Exit {float(trade['exit_price']):.6g}\n"
            f"Equity: ${equity:,.2f}\n\n"
            f"📓 <i>{_trim(review_lessons, 400)}</i>"
        )
        await self.send(text)

    async def system_event(self, text: str) -> None:
        await self.send(f"⚙️ {text}")


def _trim(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"
