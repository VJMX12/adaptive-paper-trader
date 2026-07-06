"""Configuration loading: YAML file + environment variables for secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    pass


@dataclass
class Secrets:
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    anthropic_api_key: str | None = None
    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None


@dataclass
class AppConfig:
    raw: dict[str, Any] = field(default_factory=dict)
    secrets: Secrets = field(default_factory=Secrets)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, path: str, default: Any = None) -> Any:
        """Dot-path getter: cfg.get('risk.base_risk_pct')."""
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    secrets = Secrets(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        bybit_api_key=os.getenv("BYBIT_API_KEY"),
        bybit_api_secret=os.getenv("BYBIT_API_SECRET"),
    )

    cfg = AppConfig(raw=raw, secrets=secrets)
    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig) -> None:
    required = [
        "exchange.id",
        "exchange.symbols",
        "exchange.timeframe",
        "risk.starting_equity",
        "database.path",
    ]
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ConfigError(f"Missing required config keys: {missing}")
    if cfg.get("telegram.enabled") and not (
        cfg.secrets.telegram_bot_token and cfg.secrets.telegram_chat_id
    ):
        # Not fatal: run with Telegram disabled rather than crash.
        cfg.raw.setdefault("telegram", {})["enabled"] = False
