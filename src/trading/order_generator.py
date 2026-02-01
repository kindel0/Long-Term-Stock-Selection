"""
Order generation for portfolio rebalancing.

Generates orders to transition from current holdings to target portfolio.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .fee_calculator import FeeCalculator

logger = logging.getLogger(__name__)


@dataclass
class ProposedOrder:
    """
    A proposed order pending user approval.

    Contains all information needed for review and execution.
    Supports fractional shares for IBKR.
    """

    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: float  # Supports fractional shares
    estimated_price: float
    estimated_value: float = field(default=0.0)
    estimated_fee: float = field(default=0.0)
    reason: str = ""
    tax_impact: float = 0.0  # Estimated CGT impact for sells
    current_shares: float = 0  # Supports fractional shares
    target_shares: float = 0  # Supports fractional shares
    sector: str = ""
    predicted_rank: float = 0.0

    def __post_init__(self):
        """Calculate derived fields."""
        self.estimated_value = self.quantity * self.estimated_price

    @property
    def net_value(self) -> float:
        """Net value after fees."""
        if self.action == "SELL":
            return self.estimated_value - self.estimated_fee
        else:
            return -(self.estimated_value + self.estimated_fee)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.estimated_price,
            "value": self.estimated_value,
            "fee": self.estimated_fee,
            "reason": self.reason,
            "tax_impact": self.tax_impact,
            "net_value": self.net_value,
        }


class OrderGenerator:
    """
    Generates orders for portfolio rebalancing.

    Takes current positions and target portfolio, generates
    the minimal set of orders to achieve the rebalance.

    Example:
        generator = OrderGenerator()
        orders = generator.generate_rebalance_orders(
            current_positions=current_df,
            target_portfolio=target_df,
            account_value=30000,
        )
    """

    def __init__(self, fee_calculator: Optional[FeeCalculator] = None):
        """
        Initialize the order generator.

        Args:
            fee_calculator: Calculator for estimating fees
        """
        self.fee_calc = fee_calculator or FeeCalculator()

    def generate_rebalance_orders(
        self,
        current_positions: pd.DataFrame,
        target_portfolio: pd.DataFrame,
        account_value: float,
        prices: Optional[Dict[str, float]] = None,
        min_trade_value: float = 100.0,
    ) -> List[ProposedOrder]:
        """
        Generate orders to rebalance from current to target portfolio.

        Args:
            current_positions: DataFrame with 'symbol' and 'shares' columns
            target_portfolio: DataFrame with 'TICKER', 'predicted_rank', etc.
            account_value: Total account value for position sizing
            prices: Dict of symbol -> current price (fetched if not provided)
            min_trade_value: Minimum trade value to generate

        Returns:
            List of ProposedOrder objects
        """
        orders = []

        # Get current holdings as dict
        current = {}
        if not current_positions.empty:
            if "symbol" in current_positions.columns:
                current = dict(zip(current_positions["symbol"], current_positions["shares"]))
            elif "TICKER" in current_positions.columns:
                current = dict(zip(current_positions["TICKER"], current_positions["shares"]))

        # Get target symbols
        target_symbols = set(target_portfolio["TICKER"].values) if not target_portfolio.empty else set()
        current_symbols = set(current.keys())

        # Calculate target shares for each stock (supports fractional shares)
        n_stocks = len(target_portfolio) if not target_portfolio.empty else 0
        equal_weight_value = account_value / n_stocks if n_stocks > 0 else 0

        target_shares = {}
        if not target_portfolio.empty and prices:
            for _, row in target_portfolio.iterrows():
                symbol = row["TICKER"]
                price = prices.get(symbol, 0)
                if price > 0:
                    # Round to 4 decimal places for fractional shares
                    target_shares[symbol] = round(equal_weight_value / price, 4)

        # Generate SELL orders for stocks to exit
        for symbol in current_symbols - target_symbols:
            shares = current.get(symbol, 0)
            if shares <= 0:
                continue

            price = prices.get(symbol, 0) if prices else 0

            order = ProposedOrder(
                symbol=symbol,
                action="SELL",
                quantity=shares,
                estimated_price=price,
                estimated_fee=self.fee_calc.calculate_fee(shares, price) if price > 0 else 0,
                reason="Exit position (not in target portfolio)",
                current_shares=shares,
                target_shares=0,
            )
            orders.append(order)

        # Generate orders for target positions
        for symbol in target_symbols:
            current_qty = current.get(symbol, 0)
            target_qty = target_shares.get(symbol, 0)
            price = prices.get(symbol, 0) if prices else 0

            if price <= 0:
                logger.warning(f"No price for {symbol}, skipping")
                continue

            diff = target_qty - current_qty

            if diff == 0:
                continue

            trade_value = abs(diff) * price
            if trade_value < min_trade_value:
                continue

            action = "BUY" if diff > 0 else "SELL"
            quantity = abs(diff)

            # Get additional info from target portfolio
            row = target_portfolio[target_portfolio["TICKER"] == symbol].iloc[0]
            sector = row.get("sector", "") if "sector" in row.index else ""
            pred_rank = row.get("predicted_rank", 0) if "predicted_rank" in row.index else 0

            reason = "New position" if current_qty == 0 else f"Rebalance ({action.lower()})"

            order = ProposedOrder(
                symbol=symbol,
                action=action,
                quantity=quantity,
                estimated_price=price,
                estimated_fee=self.fee_calc.calculate_fee(quantity, price),
                reason=reason,
                current_shares=current_qty,
                target_shares=target_qty,
                sector=sector,
                predicted_rank=pred_rank,
            )
            orders.append(order)

        # Sort: sells first (to free up cash), then buys by predicted rank
        orders.sort(key=lambda x: (0 if x.action == "SELL" else 1, -x.predicted_rank))

        logger.info(
            f"Generated {len(orders)} orders: "
            f"{sum(1 for o in orders if o.action == 'SELL')} sells, "
            f"{sum(1 for o in orders if o.action == 'BUY')} buys"
        )

        return orders

    def generate_initial_orders(
        self,
        target_portfolio: pd.DataFrame,
        account_value: float,
        prices: Dict[str, float],
        reserve_pct: float = 0.02,
    ) -> List[ProposedOrder]:
        """
        Generate orders for initial portfolio construction.

        Args:
            target_portfolio: Target stocks with 'TICKER' and 'predicted_rank'
            account_value: Total cash available
            prices: Current prices for all symbols
            reserve_pct: Fraction of cash to reserve for fees/buffer

        Returns:
            List of ProposedOrder objects (all buys)
        """
        orders = []

        deployable = account_value * (1 - reserve_pct)
        n_stocks = len(target_portfolio)

        if n_stocks == 0:
            return orders

        per_stock = deployable / n_stocks

        for _, row in target_portfolio.iterrows():
            symbol = row["TICKER"]
            price = prices.get(symbol, 0)

            if price <= 0:
                logger.warning(f"No price for {symbol}, skipping")
                continue

            # Support fractional shares (round to 4 decimal places)
            shares = round(per_stock / price, 4)

            if shares * price < 1.0:  # IBKR minimum order is $1
                logger.warning(f"Order value too small for {symbol}, skipping")
                continue

            order = ProposedOrder(
                symbol=symbol,
                action="BUY",
                quantity=shares,
                estimated_price=price,
                estimated_fee=self.fee_calc.calculate_fee(shares, price),
                reason="Initial position",
                current_shares=0,
                target_shares=shares,
                sector=row.get("sector", "") if "sector" in row.index else "",
                predicted_rank=row.get("predicted_rank", 0) if "predicted_rank" in row.index else 0,
            )
            orders.append(order)

        # Sort by predicted rank (highest first)
        orders.sort(key=lambda x: -x.predicted_rank)

        return orders

    def estimate_order_costs(self, orders: List[ProposedOrder]) -> Dict:
        """
        Estimate total costs for a list of orders.

        Args:
            orders: List of ProposedOrder objects

        Returns:
            Dict with cost breakdown
        """
        total_fees = sum(o.estimated_fee for o in orders)
        total_buys = sum(o.estimated_value for o in orders if o.action == "BUY")
        total_sells = sum(o.estimated_value for o in orders if o.action == "SELL")

        return {
            "total_fees": total_fees,
            "total_buys": total_buys,
            "total_sells": total_sells,
            "net_cash_flow": total_sells - total_buys - total_fees,
            "n_buys": sum(1 for o in orders if o.action == "BUY"),
            "n_sells": sum(1 for o in orders if o.action == "SELL"),
        }

    def to_dataframe(self, orders: List[ProposedOrder]) -> pd.DataFrame:
        """Convert orders to DataFrame for display/export."""
        if not orders:
            return pd.DataFrame()

        return pd.DataFrame([o.to_dict() for o in orders])

    def print_orders(self, orders: List[ProposedOrder]) -> None:
        """Print orders in a formatted table."""
        if not orders:
            print("No orders to display")
            return

        print("\n" + "=" * 80)
        print("PROPOSED ORDERS")
        print("=" * 80)

        # Sells
        sells = [o for o in orders if o.action == "SELL"]
        if sells:
            print("\nSELLS:")
            print(f"{'Symbol':<10} {'Shares':>10} {'Price':>10} {'Value':>12} {'Fee':>8} {'Reason'}")
            print("-" * 70)
            for o in sells:
                print(
                    f"{o.symbol:<10} {o.quantity:>10} "
                    f"${o.estimated_price:>9.2f} ${o.estimated_value:>11.2f} "
                    f"${o.estimated_fee:>7.2f} {o.reason}"
                )

        # Buys
        buys = [o for o in orders if o.action == "BUY"]
        if buys:
            print("\nBUYS:")
            print(f"{'Symbol':<10} {'Shares':>10} {'Price':>10} {'Value':>12} {'Fee':>8} {'Reason'}")
            print("-" * 70)
            for o in buys:
                print(
                    f"{o.symbol:<10} {o.quantity:>10} "
                    f"${o.estimated_price:>9.2f} ${o.estimated_value:>11.2f} "
                    f"${o.estimated_fee:>7.2f} {o.reason}"
                )

        # Summary
        costs = self.estimate_order_costs(orders)
        print("\n" + "-" * 80)
        print("SUMMARY:")
        print(f"  Sells: {costs['n_sells']} orders, ${costs['total_sells']:.2f}")
        print(f"  Buys:  {costs['n_buys']} orders, ${costs['total_buys']:.2f}")
        print(f"  Fees:  ${costs['total_fees']:.2f}")
        print(f"  Net Cash Flow: ${costs['net_cash_flow']:.2f}")
        print("=" * 80)
