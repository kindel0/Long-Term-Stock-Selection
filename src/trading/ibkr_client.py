"""
Interactive Brokers API client wrapper.

Provides a clean interface for IBKR operations using ib_insync.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ..config import IBKR_PORTS

logger = logging.getLogger(__name__)


@dataclass
class AccountSummary:
    """Summary of IBKR account."""
    net_liquidation: float
    total_cash: float
    buying_power: float
    gross_position_value: float
    maintenance_margin: float
    available_funds: float
    currency: str = "USD"


class IBKRClient:
    """
    Interactive Brokers API client.

    Wraps ib_insync for cleaner interface. Supports both
    paper and live trading modes.

    Example:
        client = IBKRClient(mode='paper')
        await client.connect()
        positions = await client.get_positions()
        await client.disconnect()
    """

    def __init__(self, mode: str = "paper", client_id: int = 1):
        """
        Initialize the IBKR client.

        Args:
            mode: 'paper' or 'live'
            client_id: Client ID for connection
        """
        self.mode = mode
        self.port = IBKR_PORTS.get(mode, 7497)
        self.client_id = client_id
        self.ib = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to TWS/Gateway."""
        if self.ib is None:
            return False
        return self.ib.isConnected()

    async def connect(self, host: str = "127.0.0.1") -> bool:
        """
        Connect to TWS or IB Gateway.

        Args:
            host: Host address (usually localhost)

        Returns:
            True if connected successfully
        """
        try:
            from ib_insync import IB
        except ImportError:
            logger.error("ib_insync not installed. Install with: pip install ib_insync")
            return False

        self.ib = IB()

        try:
            await self.ib.connectAsync(host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(f"Connected to IBKR ({self.mode} mode) on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    async def get_account_summary(self) -> Optional[AccountSummary]:
        """
        Get account summary.

        Returns:
            AccountSummary object or None if not connected
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR")
            return None

        try:
            summary = await self.ib.accountSummaryAsync()

            values = {item.tag: float(item.value) for item in summary}

            return AccountSummary(
                net_liquidation=values.get("NetLiquidation", 0),
                total_cash=values.get("TotalCashValue", 0),
                buying_power=values.get("BuyingPower", 0),
                gross_position_value=values.get("GrossPositionValue", 0),
                maintenance_margin=values.get("MaintMarginReq", 0),
                available_funds=values.get("AvailableFunds", 0),
            )
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return None

    async def get_positions(self) -> pd.DataFrame:
        """
        Get current positions.

        Returns:
            DataFrame with position details
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR")
            return pd.DataFrame()

        try:
            positions = await self.ib.positionsAsync()

            if not positions:
                return pd.DataFrame()

            data = []
            for pos in positions:
                data.append({
                    "symbol": pos.contract.symbol,
                    "shares": pos.position,
                    "avg_cost": pos.avgCost,
                    "market_value": pos.position * pos.avgCost,  # Approximate
                    "contract_type": pos.contract.secType,
                    "exchange": pos.contract.exchange,
                })

            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return pd.DataFrame()

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current price or None
        """
        if not self.is_connected:
            return None

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            ticker = self.ib.reqMktData(contract)
            await self.ib.sleep(1)  # Wait for data

            price = ticker.marketPrice()
            self.ib.cancelMktData(contract)

            return price if price > 0 else None
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    async def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict of symbol -> price
        """
        prices = {}
        for symbol in symbols:
            price = await self.get_current_price(symbol)
            if price:
                prices[symbol] = price
        return prices

    async def place_market_order(
        self, symbol: str, quantity: int, action: str
    ) -> Optional[str]:
        """
        Place a market order.

        Args:
            symbol: Stock ticker
            quantity: Number of shares
            action: 'BUY' or 'SELL'

        Returns:
            Order ID or None if failed
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR")
            return None

        try:
            from ib_insync import Stock, MarketOrder

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)

            logger.info(f"Placed {action} order: {quantity} {symbol}")
            return str(trade.order.orderId)
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def place_limit_order(
        self, symbol: str, quantity: int, action: str, limit_price: float
    ) -> Optional[str]:
        """
        Place a limit order.

        Args:
            symbol: Stock ticker
            quantity: Number of shares
            action: 'BUY' or 'SELL'
            limit_price: Limit price

        Returns:
            Order ID or None if failed
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR")
            return None

        try:
            from ib_insync import Stock, LimitOrder

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            order = LimitOrder(action, quantity, limit_price)
            trade = self.ib.placeOrder(contract, order)

            logger.info(f"Placed {action} limit order: {quantity} {symbol} @ {limit_price}")
            return str(trade.order.orderId)
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if not self.is_connected:
            return False

        try:
            orders = self.ib.orders()
            for order in orders:
                if str(order.orderId) == order_id:
                    self.ib.cancelOrder(order)
                    logger.info(f"Cancelled order {order_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    async def get_open_orders(self) -> pd.DataFrame:
        """Get all open orders."""
        if not self.is_connected:
            return pd.DataFrame()

        try:
            orders = self.ib.openOrders()
            if not orders:
                return pd.DataFrame()

            data = []
            for order in orders:
                data.append({
                    "order_id": order.orderId,
                    "symbol": order.contract.symbol if hasattr(order, "contract") else "",
                    "action": order.action,
                    "quantity": order.totalQuantity,
                    "order_type": order.orderType,
                    "status": order.status,
                })

            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return pd.DataFrame()

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
    ) -> pd.DataFrame:
        """
        Get historical price data.

        Args:
            symbol: Stock ticker
            duration: Time span (e.g., '1 Y', '6 M')
            bar_size: Bar size (e.g., '1 day', '1 hour')

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected:
            return pd.DataFrame()

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="ADJUSTED_LAST",
                useRTH=True,
            )

            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame(bars)
            df["date"] = pd.to_datetime(df["date"])
            return df

        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
