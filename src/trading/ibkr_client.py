"""
Interactive Brokers API client wrapper.

Provides a clean interface for IBKR operations using ib_insync.
Uses synchronous API to avoid event loop conflicts.
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

    Uses synchronous API to avoid event loop conflicts.

    Example:
        client = IBKRClient(mode='paper')
        client.connect()
        positions = client.get_positions()
        client.disconnect()
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

    def connect(self, host: str = "127.0.0.1") -> bool:
        """
        Connect to TWS or IB Gateway.

        Args:
            host: Host address (usually localhost)

        Returns:
            True if connected successfully
        """
        try:
            from ib_insync import IB, util
            # Enable nested asyncio for ib_insync
            util.startLoop()
        except ImportError:
            logger.error("ib_insync not installed. Install with: pip install ib_insync")
            return False

        self.ib = IB()

        try:
            self.ib.connect(host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(f"Connected to IBKR ({self.mode} mode) on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def get_account_summary(self) -> Optional[AccountSummary]:
        """
        Get account summary.

        Returns:
            AccountSummary object or None if not connected
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR")
            return None

        try:
            # Request account values
            self.ib.reqAccountSummary()
            self.ib.sleep(1)  # Wait for data

            summary = self.ib.accountSummary()

            # Parse values, handling non-numeric fields
            values = {}
            for item in summary:
                try:
                    values[item.tag] = float(item.value)
                except (ValueError, TypeError):
                    pass  # Skip non-numeric fields like 'INDIVIDUAL'

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

    def get_positions(self) -> pd.DataFrame:
        """
        Get current positions.

        Returns:
            DataFrame with position details
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR")
            return pd.DataFrame()

        try:
            positions = self.ib.positions()

            if not positions:
                return pd.DataFrame()

            data = []
            for pos in positions:
                data.append({
                    "symbol": pos.contract.symbol,
                    "shares": pos.position,
                    "avg_cost": pos.avgCost,
                    "market_value": pos.position * pos.avgCost,
                    "contract_type": pos.contract.secType,
                    "exchange": pos.contract.exchange,
                })

            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
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

            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)  # Wait for data

            # Try different price fields
            price = None
            if ticker.last and ticker.last > 0:
                price = ticker.last
            elif ticker.close and ticker.close > 0:
                price = ticker.close
            elif ticker.bid and ticker.ask:
                price = (ticker.bid + ticker.ask) / 2

            self.ib.cancelMktData(contract)

            return price
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict of symbol -> price
        """
        if not self.is_connected:
            return {}

        try:
            from ib_insync import Stock

            # Create contracts for all symbols
            contracts = [Stock(sym, "SMART", "USD") for sym in symbols]
            self.ib.qualifyContracts(*contracts)

            # Request market data for all
            tickers = []
            for contract in contracts:
                ticker = self.ib.reqMktData(contract, '', False, False)
                tickers.append((contract.symbol, ticker))

            # Wait for data
            self.ib.sleep(3)

            # Collect prices
            prices = {}
            for symbol, ticker in tickers:
                price = None
                if ticker.last and ticker.last > 0:
                    price = ticker.last
                elif ticker.close and ticker.close > 0:
                    price = ticker.close
                elif ticker.bid and ticker.ask and ticker.bid > 0:
                    price = (ticker.bid + ticker.ask) / 2

                if price:
                    prices[symbol] = price

                self.ib.cancelMktData(ticker.contract)

            return prices
        except Exception as e:
            logger.error(f"Failed to get prices: {e}")
            return {}

    def place_market_order(
        self, symbol: str, quantity: float, action: str
    ) -> Optional[str]:
        """
        Place a market order. Supports fractional shares.

        Args:
            symbol: Stock ticker
            quantity: Number of shares (can be fractional, e.g., 1.5)
            action: 'BUY' or 'SELL'

        Returns:
            Order ID or None if failed

        Note:
            Fractional shares on IBKR require:
            - US stocks only
            - Market orders only
            - Regular trading hours (9:30 AM - 4:00 PM ET)
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR")
            return None

        try:
            from ib_insync import Stock, MarketOrder

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # IBKR accepts fractional quantities directly
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)

            # Wait for order to be submitted
            self.ib.sleep(1)

            qty_str = f"{quantity:.4f}" if quantity % 1 else f"{int(quantity)}"
            logger.info(f"Placed {action} order: {qty_str} {symbol}, orderId={trade.order.orderId}")
            return str(trade.order.orderId)
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            return None

    def place_limit_order(
        self, symbol: str, quantity: float, action: str, limit_price: float
    ) -> Optional[str]:
        """
        Place a limit order. Note: Fractional shares do NOT support limit orders.

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

            self.ib.sleep(1)

            logger.info(f"Placed {action} limit order: {quantity} {symbol} @ {limit_price}")
            return str(trade.order.orderId)
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def wait_for_fill(self, order_id: str, timeout: int = 30) -> Optional[float]:
        """
        Wait for an order to fill.

        Args:
            order_id: Order ID to wait for
            timeout: Maximum seconds to wait

        Returns:
            Fill price or None if not filled
        """
        if not self.is_connected:
            return None

        try:
            # Find the trade
            for trade in self.ib.trades():
                if str(trade.order.orderId) == order_id:
                    # Wait for fill
                    start = datetime.now()
                    while (datetime.now() - start).seconds < timeout:
                        self.ib.sleep(0.5)
                        if trade.orderStatus.status == 'Filled':
                            return trade.orderStatus.avgFillPrice
                        elif trade.orderStatus.status in ('Cancelled', 'ApiCancelled'):
                            return None
                    return None
            return None
        except Exception as e:
            logger.error(f"Error waiting for fill: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
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
            for trade in self.ib.trades():
                if str(trade.order.orderId) == order_id:
                    self.ib.cancelOrder(trade.order)
                    self.ib.sleep(1)
                    logger.info(f"Cancelled order {order_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def get_open_orders(self) -> pd.DataFrame:
        """Get all open orders."""
        if not self.is_connected:
            return pd.DataFrame()

        try:
            trades = self.ib.openTrades()
            if not trades:
                return pd.DataFrame()

            data = []
            for trade in trades:
                data.append({
                    "order_id": trade.order.orderId,
                    "symbol": trade.contract.symbol,
                    "action": trade.order.action,
                    "quantity": trade.order.totalQuantity,
                    "order_type": trade.order.orderType,
                    "status": trade.orderStatus.status,
                })

            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return pd.DataFrame()

    def get_historical_data(
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

            bars = self.ib.reqHistoricalData(
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
