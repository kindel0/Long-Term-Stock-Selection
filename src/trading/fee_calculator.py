"""
IBKR fee calculation.

Calculates trading fees based on IBKR's fee structure.
Supports both tiered and fixed pricing models.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from ..config import IBKR_FEES, DEFAULT_IBKR_PRICING

logger = logging.getLogger(__name__)


@dataclass
class FeeBreakdown:
    """Detailed breakdown of trading fees."""

    base_fee: float
    exchange_fees: float
    regulatory_fees: float
    total: float
    fee_as_pct: float


class FeeCalculator:
    """
    Calculates IBKR trading fees.

    Supports tiered and fixed pricing for US stocks.
    Default is tiered pricing which is typically better
    for smaller accounts.

    Example:
        calc = FeeCalculator()
        fee = calc.calculate_fee(100, 50.00)  # 100 shares at $50
        print(f"Fee: ${fee:.2f}")
    """

    def __init__(self, pricing_model: str = None):
        """
        Initialize the fee calculator.

        Args:
            pricing_model: 'tiered' or 'fixed' (default from config)
        """
        self.pricing_model = pricing_model or DEFAULT_IBKR_PRICING
        self.fee_structure = IBKR_FEES.get(self.pricing_model, IBKR_FEES["tiered"])

    def calculate_fee(
        self,
        shares: int,
        price: float,
        include_exchange: bool = True,
        removing_liquidity: bool = True,
    ) -> float:
        """
        Calculate the total fee for a trade.

        Args:
            shares: Number of shares
            price: Price per share
            include_exchange: Whether to include exchange fees
            removing_liquidity: True for market orders (removes liquidity)

        Returns:
            Total fee in USD
        """
        if shares <= 0 or price <= 0:
            return 0.0

        us_fees = self.fee_structure.get("us_stocks", {})
        trade_value = shares * price

        # Base commission
        per_share = us_fees.get("per_share", 0.0035)
        min_fee = us_fees.get("min_per_order", 0.35)
        max_pct = us_fees.get("max_pct", 0.01)

        base_fee = shares * per_share
        base_fee = max(base_fee, min_fee)
        base_fee = min(base_fee, trade_value * max_pct)

        # Exchange fees (simplified estimate)
        exchange_fee = 0.0
        if include_exchange and self.pricing_model == "tiered":
            exchange_fees = self.fee_structure.get("exchange_fees", {})
            if removing_liquidity:
                exchange_fee = shares * exchange_fees.get("remove_liquidity", 0.003)
            else:
                # Rebate for adding liquidity
                exchange_fee = shares * exchange_fees.get("add_liquidity", -0.002)

        total = base_fee + exchange_fee

        return max(0, total)

    def calculate_fee_breakdown(
        self,
        shares: int,
        price: float,
        removing_liquidity: bool = True,
    ) -> FeeBreakdown:
        """
        Get a detailed breakdown of fees.

        Args:
            shares: Number of shares
            price: Price per share
            removing_liquidity: True for market orders

        Returns:
            FeeBreakdown with component details
        """
        if shares <= 0 or price <= 0:
            return FeeBreakdown(0, 0, 0, 0, 0)

        us_fees = self.fee_structure.get("us_stocks", {})
        trade_value = shares * price

        # Base commission
        per_share = us_fees.get("per_share", 0.0035)
        min_fee = us_fees.get("min_per_order", 0.35)
        max_pct = us_fees.get("max_pct", 0.01)

        base_fee = shares * per_share
        base_fee = max(base_fee, min_fee)
        base_fee = min(base_fee, trade_value * max_pct)

        # Exchange fees
        exchange_fee = 0.0
        if self.pricing_model == "tiered":
            exchange_fees = self.fee_structure.get("exchange_fees", {})
            if removing_liquidity:
                exchange_fee = shares * exchange_fees.get("remove_liquidity", 0.003)
            else:
                exchange_fee = shares * exchange_fees.get("add_liquidity", -0.002)

        # Regulatory fees (rough estimate)
        sec_fee = trade_value * 0.0000278  # SEC fee
        finra_fee = min(shares * 0.000145, 7.27)  # FINRA TAF
        regulatory = sec_fee + finra_fee

        total = base_fee + exchange_fee + regulatory
        total = max(0, total)

        fee_as_pct = (total / trade_value * 100) if trade_value > 0 else 0

        return FeeBreakdown(
            base_fee=base_fee,
            exchange_fees=exchange_fee,
            regulatory_fees=regulatory,
            total=total,
            fee_as_pct=fee_as_pct,
        )

    def estimate_rebalance_cost(
        self,
        trades: list,
        removing_liquidity: bool = True,
    ) -> Dict:
        """
        Estimate total cost for a rebalance.

        Args:
            trades: List of (shares, price) tuples
            removing_liquidity: True for market orders

        Returns:
            Dict with cost breakdown
        """
        total_fees = 0.0
        total_value = 0.0
        trade_count = 0

        for shares, price in trades:
            if shares == 0:
                continue

            fee = self.calculate_fee(abs(shares), price, removing_liquidity=removing_liquidity)
            total_fees += fee
            total_value += abs(shares) * price
            trade_count += 1

        return {
            "total_fees": total_fees,
            "total_traded_value": total_value,
            "trade_count": trade_count,
            "avg_fee_per_trade": total_fees / trade_count if trade_count > 0 else 0,
            "fee_as_pct": (total_fees / total_value * 100) if total_value > 0 else 0,
        }

    def estimate_quarterly_costs(
        self,
        portfolio_value: float,
        n_stocks: int,
        avg_turnover_pct: float = 0.5,
        avg_share_price: float = 100.0,
    ) -> Dict:
        """
        Estimate quarterly trading costs.

        Args:
            portfolio_value: Total portfolio value
            n_stocks: Number of stocks in portfolio
            avg_turnover_pct: Fraction of portfolio traded each quarter
            avg_share_price: Average price per share

        Returns:
            Dict with estimated costs
        """
        # Value traded
        value_traded = portfolio_value * avg_turnover_pct

        # Estimate number of trades (buys + sells)
        # Assume we sell half the stocks and buy half
        n_trades = int(n_stocks * avg_turnover_pct * 2)

        # Average trade size
        avg_trade_value = value_traded / n_trades if n_trades > 0 else 0
        avg_shares = int(avg_trade_value / avg_share_price) if avg_share_price > 0 else 0

        # Calculate fees
        trades = [(avg_shares, avg_share_price)] * n_trades
        costs = self.estimate_rebalance_cost(trades)

        costs["portfolio_value"] = portfolio_value
        costs["turnover_pct"] = avg_turnover_pct
        costs["estimated_trades"] = n_trades
        costs["annualized_cost"] = costs["total_fees"] * 4

        return costs
