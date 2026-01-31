"""
Position management and tracking.

Tracks current portfolio positions with cost basis and P&L.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import PORTFOLIOS_DIR

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single position in the portfolio."""

    symbol: str
    shares: int
    avg_cost: float
    total_cost: float = field(default=0.0)
    current_price: float = field(default=0.0)
    market_value: float = field(default=0.0)
    unrealized_pnl: float = field(default=0.0)
    unrealized_pnl_pct: float = field(default=0.0)
    sector: str = ""
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Calculate derived fields."""
        if self.total_cost == 0.0:
            self.total_cost = self.shares * self.avg_cost
        self.update_market_value(self.current_price)

    def update_market_value(self, price: float) -> None:
        """Update market value and P&L with new price."""
        self.current_price = price
        self.market_value = self.shares * price
        self.unrealized_pnl = self.market_value - self.total_cost
        if self.total_cost > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / self.total_cost) * 100
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class PositionManager:
    """
    Manages portfolio positions.

    Tracks positions, updates with fills, and calculates P&L.

    Example:
        pm = PositionManager()
        pm.add_position("AAPL", 100, 150.00)
        pm.update_prices({"AAPL": 160.00})
        print(pm.get_summary())
    """

    def __init__(self, portfolio_dir: Optional[Path] = None):
        """
        Initialize the position manager.

        Args:
            portfolio_dir: Directory for saving portfolio snapshots
        """
        self.portfolio_dir = Path(portfolio_dir) if portfolio_dir else PORTFOLIOS_DIR
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)

        self.positions: Dict[str, Position] = {}
        self.cash: float = 0.0
        self.last_updated: Optional[datetime] = None

    def add_position(
        self,
        symbol: str,
        shares: int,
        price: float,
        sector: str = "",
    ) -> Position:
        """
        Add or update a position after a buy.

        Args:
            symbol: Ticker symbol
            shares: Shares bought
            price: Purchase price per share
            sector: Optional sector classification

        Returns:
            Updated Position object
        """
        if symbol in self.positions:
            # Update existing position (average cost)
            pos = self.positions[symbol]
            new_total_cost = pos.total_cost + (shares * price)
            new_shares = pos.shares + shares
            pos.shares = new_shares
            pos.total_cost = new_total_cost
            pos.avg_cost = new_total_cost / new_shares if new_shares > 0 else 0
            pos.update_market_value(price)
        else:
            # New position
            pos = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=price,
                current_price=price,
                sector=sector,
            )
            self.positions[symbol] = pos

        self.last_updated = datetime.now()
        return pos

    def reduce_position(
        self, symbol: str, shares: int, price: float
    ) -> Optional[Position]:
        """
        Reduce a position after a sell.

        Args:
            symbol: Ticker symbol
            shares: Shares sold
            price: Sale price per share

        Returns:
            Updated Position or None if position closed
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot reduce {symbol}: position not found")
            return None

        pos = self.positions[symbol]

        if shares >= pos.shares:
            # Close entire position
            del self.positions[symbol]
            return None

        # Reduce position (keep same avg cost)
        pos.shares -= shares
        pos.total_cost = pos.shares * pos.avg_cost
        pos.update_market_value(price)

        self.last_updated = datetime.now()
        return pos

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update all positions with current prices.

        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_market_value(prices[symbol])

        self.last_updated = datetime.now()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a single position."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all positions as a list."""
        return list(self.positions.values())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert positions to DataFrame."""
        if not self.positions:
            return pd.DataFrame()

        return pd.DataFrame([p.to_dict() for p in self.positions.values()])

    def get_summary(self) -> Dict:
        """
        Get portfolio summary statistics.

        Returns:
            Dict with portfolio metrics
        """
        if not self.positions:
            return {
                "total_value": self.cash,
                "cash": self.cash,
                "invested": 0,
                "n_positions": 0,
                "unrealized_pnl": 0,
                "unrealized_pnl_pct": 0,
            }

        total_value = sum(p.market_value for p in self.positions.values())
        total_cost = sum(p.total_cost for p in self.positions.values())
        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        return {
            "total_value": total_value + self.cash,
            "cash": self.cash,
            "invested": total_value,
            "total_cost": total_cost,
            "n_positions": len(self.positions),
            "unrealized_pnl": total_pnl,
            "unrealized_pnl_pct": (total_pnl / total_cost * 100) if total_cost > 0 else 0,
            "largest_position": max(
                (p.market_value for p in self.positions.values()), default=0
            ),
            "smallest_position": min(
                (p.market_value for p in self.positions.values()), default=0
            ),
        }

    def get_sector_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation by sector."""
        allocations = {}
        total = sum(p.market_value for p in self.positions.values())

        if total == 0:
            return {}

        for pos in self.positions.values():
            sector = pos.sector or "Unknown"
            allocations[sector] = allocations.get(sector, 0) + pos.market_value

        # Convert to percentages
        return {k: v / total * 100 for k, v in allocations.items()}

    def save_snapshot(self, name: Optional[str] = None) -> Path:
        """
        Save current portfolio state.

        Args:
            name: Optional snapshot name (uses timestamp if not provided)

        Returns:
            Path to saved file
        """
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")

        filepath = self.portfolio_dir / f"portfolio_{name}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "cash": self.cash,
            "positions": [p.to_dict() for p in self.positions.values()],
            "summary": self.get_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved portfolio snapshot: {filepath}")
        return filepath

    def load_snapshot(self, filepath: Path) -> None:
        """
        Load portfolio state from file.

        Args:
            filepath: Path to snapshot file
        """
        with open(filepath) as f:
            data = json.load(f)

        self.cash = data.get("cash", 0)
        self.positions = {}

        for pos_data in data.get("positions", []):
            pos = Position(**pos_data)
            self.positions[pos.symbol] = pos

        self.last_updated = datetime.now()
        logger.info(f"Loaded portfolio: {len(self.positions)} positions")

    def print_summary(self) -> None:
        """Print portfolio summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"Total Value:    ${summary['total_value']:,.2f}")
        print(f"  Invested:     ${summary['invested']:,.2f}")
        print(f"  Cash:         ${summary['cash']:,.2f}")
        print(f"Positions:      {summary['n_positions']}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f} ({summary['unrealized_pnl_pct']:.1f}%)")

        if self.positions:
            print("\n" + "-" * 60)
            print(f"{'Symbol':<10} {'Shares':>8} {'Price':>10} {'Value':>12} {'P&L':>10} {'%':>8}")
            print("-" * 60)

            for pos in sorted(self.positions.values(), key=lambda x: -x.market_value):
                print(
                    f"{pos.symbol:<10} {pos.shares:>8} "
                    f"${pos.current_price:>9.2f} ${pos.market_value:>11.2f} "
                    f"${pos.unrealized_pnl:>9.2f} {pos.unrealized_pnl_pct:>7.1f}%"
                )

        print("=" * 60)
