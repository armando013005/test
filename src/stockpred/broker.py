"""Alpaca paper-trading client. It refuses to connect to a live account."""

from __future__ import annotations

import logging
import os

from .live import Order

log = logging.getLogger(__name__)

LIVE_HOST = "api.alpaca.markets"


class LiveTradingRefused(RuntimeError):
    pass


def _check_paper_only() -> None:
    for var in ("APCA_API_BASE_URL", "ALPACA_BASE_URL"):
        url = os.environ.get(var, "")
        if url and "paper" not in url:
            raise LiveTradingRefused(f"{var}={url} is not a paper endpoint. This tool trades paper accounts only.")


class AlpacaPaperBroker:
    """Thin wrapper over alpaca-py. Keys come from ALPACA_API_KEY and ALPACA_SECRET_KEY."""

    def __init__(self) -> None:
        _check_paper_only()
        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to your paper account keys.")
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(key, secret, paper=True)
        base = str(getattr(self.client, "_base_url", ""))
        if LIVE_HOST in base and "paper" not in base:
            raise LiveTradingRefused(f"Client resolved to live endpoint {base}.")

    def account(self) -> tuple[float, float]:
        acct = self.client.get_account()
        return float(acct.equity), float(acct.cash)

    def positions(self) -> dict[str, int]:
        return {p.symbol.replace(".", "-"): int(float(p.qty)) for p in self.client.get_all_positions()}

    def _open_stop_orders(self, symbol: str):
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], nested=True)
        )
        stops = []
        for o in orders:
            for leg in [o, *(o.legs or [])]:
                if leg.side == OrderSide.SELL and leg.stop_price is not None:
                    stops.append(leg)
        return stops

    def submit(self, order: Order) -> None:
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
            )
            self.client.submit_order(req)
        elif order.action == "sell":
            for stop in self._open_stop_orders(symbol):
                self.client.cancel_order_by_id(stop.id)
            self.client.close_position(symbol)
        elif order.action == "update_stop":
            stops = self._open_stop_orders(symbol)
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
                    )
                )
        else:
            raise ValueError(f"Unknown action {order.action}")
        log.info("Submitted %s %s x%d stop=%s", order.action, order.ticker, order.shares, order.stop)
