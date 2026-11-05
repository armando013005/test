"""Alpaca paper-trading client. It refuses to connect to a live account."""

from __future__ import annotations

import logging
import os
import time
from urllib.parse import urlparse

from .live import Order

log = logging.getLogger(__name__)

PAPER_HOST = "paper-api.alpaca.markets"


class LiveTradingRefused(RuntimeError):
    pass


def _check_paper_only() -> None:
    """Refuses environment overrides that point anywhere but the paper endpoint."""
    for var in ("APCA_API_BASE_URL", "ALPACA_BASE_URL"):
        url = os.environ.get(var, "")
        if url and urlparse(url).hostname != PAPER_HOST:
            raise LiveTradingRefused(f"{var}={url} is not a paper endpoint. This tool trades paper accounts only.")


def client_host(client) -> str | None:
    """Host the alpaca-py client will call. Its base URL is stored as an enum."""
    base = getattr(client, "_base_url", None)
    url = getattr(base, "value", base)
    return urlparse(str(url)).hostname if url else None


def have_keys() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


class AlpacaPaperBroker:
    """Thin wrapper over alpaca-py. Keys come from ALPACA_API_KEY and ALPACA_SECRET_KEY."""

    def __init__(self) -> None:
        _check_paper_only()
        if not have_keys():
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to your paper account keys.")
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
        host = client_host(self.client)
        if host != PAPER_HOST:
            raise LiveTradingRefused(f"Alpaca client resolved to {host!r}, not {PAPER_HOST}.")

    def market_is_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def account(self) -> tuple[float, float]:
        acct = self.client.get_account()
        return float(acct.equity), float(acct.cash)

    def positions(self) -> dict[str, int]:
        return {p.symbol.replace(".", "-"): int(float(p.qty)) for p in self.client.get_all_positions()}

    def open_orders(self) -> list:
        """Every open order, with bracket legs listed as their own orders."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return list(self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False, limit=500)))

    def cancel_stale_buys(self, keep_prefix: str) -> list[str]:
        """Cancels open buys from earlier runs. They missed their open and would double up."""
        from alpaca.trading.enums import OrderSide

        canceled = []
        for o in self.open_orders():
            if o.side == OrderSide.BUY and not str(o.client_order_id or "").startswith(keep_prefix):
                self.client.cancel_order_by_id(o.id)
                canceled.append(o.symbol.replace(".", "-"))
        return canceled

    def _stop_orders(self, symbol: str) -> list:
        from alpaca.trading.enums import OrderSide

        return [
            o for o in self.open_orders()
            if o.symbol == symbol and o.side == OrderSide.SELL and o.stop_price is not None
        ]

    def _wait_canceled(self, order_ids: list, timeout: float = 10.0) -> None:
        """Blocks until canceled stops release their shares, so the sell isn't rejected."""
        from alpaca.trading.enums import OrderStatus

        done = {OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
        pending = set(order_ids)
        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            pending = {oid for oid in pending if self.client.get_order_by_id(oid).status not in done}
            if pending:
                time.sleep(0.5)
        if pending:
            raise RuntimeError(f"Stop orders still open after {timeout:.0f}s: {sorted(map(str, pending))}")

    def submit(self, order: Order, client_order_id: str | None = None) -> None:
        """Sends one order. A reused client_order_id makes Alpaca reject a duplicate."""
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest,
            ReplaceOrderRequest,
            StopLossRequest,
            StopOrderRequest,
        )

        symbol = order.ticker.replace("-", ".")
        if order.action == "buy":
            req = MarketOrderRequest(
                symbol=symbol,
                qty=order.shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.OTO,
                stop_loss=StopLossRequest(stop_price=round(order.stop, 2)),
                client_order_id=client_order_id,
            )
            self.client.submit_order(req)
        elif order.action == "sell":
            stops = self._stop_orders(symbol)
            for stop in stops:
                self.client.cancel_order_by_id(stop.id)
            self._wait_canceled([stop.id for stop in stops])
            self.client.submit_order(
                MarketOrderRequest(
                    symbol=symbol, qty=order.shares, side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY, client_order_id=client_order_id,
                )
            )
        elif order.action == "update_stop":
            stops = self._stop_orders(symbol)
            if stops:
                for stop in stops:
                    self.client.replace_order_by_id(stop.id, ReplaceOrderRequest(stop_price=round(order.stop, 2)))
            else:
                self.client.submit_order(
                    StopOrderRequest(
                        symbol=symbol,
                        qty=order.shares,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        stop_price=round(order.stop, 2),
                        client_order_id=client_order_id,
                    )
                )
        else:
            raise ValueError(f"Unknown action {order.action}")
        log.info("Submitted %s %s x%d stop=%s", order.action, order.ticker, order.shares, order.stop)
