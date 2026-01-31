"""
Dividend income and withholding tax tracker.

Tracks dividend income from US stocks including:
- Gross dividend amounts
- US withholding tax (15% with W-8BEN)
- Net amounts received
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import IRELAND_TAX, TRADES_DIR

logger = logging.getLogger(__name__)


@dataclass
class DividendRecord:
    """Record of a dividend payment."""

    symbol: str
    payment_date: datetime
    shares: int
    gross_per_share: float
    gross_amount: float
    withholding_rate: float
    withholding_amount: float
    net_amount: float
    currency: str = "USD"
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "payment_date": self.payment_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DividendRecord":
        """Create from dictionary."""
        data["payment_date"] = datetime.fromisoformat(data["payment_date"])
        return cls(**data)


@dataclass
class AnnualDividendSummary:
    """Annual dividend summary for tax return."""

    year: int
    total_gross: float
    total_withholding: float
    total_net: float
    n_payments: int
    dividends_by_symbol: Dict[str, float]
    withholding_by_symbol: Dict[str, float]
    records: List[DividendRecord]


class DividendTracker:
    """
    Tracks dividend income and withholding taxes.

    For US stocks held by Irish residents:
    - Gross dividends are taxable as income in Ireland
    - US withholding tax (15% with W-8BEN) can be credited
    - Net received = Gross - US withholding

    Example:
        tracker = DividendTracker()
        tracker.record_dividend("AAPL", datetime.now(), 100, 0.25)
        summary = tracker.get_annual_summary(2024)
    """

    def __init__(
        self,
        default_withholding_rate: float = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize the dividend tracker.

        Args:
            default_withholding_rate: Default US withholding rate
            data_dir: Directory for persisting data
        """
        self.default_withholding_rate = (
            default_withholding_rate or IRELAND_TAX["us_withholding_rate"]
        )
        self.data_dir = Path(data_dir) if data_dir else TRADES_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[DividendRecord] = []

    def record_dividend(
        self,
        symbol: str,
        payment_date: datetime,
        shares: int,
        dividend_per_share: float,
        withholding_rate: float = None,
        currency: str = "USD",
        notes: str = "",
    ) -> DividendRecord:
        """
        Record a dividend payment.

        Args:
            symbol: Ticker symbol
            payment_date: Date dividend was paid
            shares: Number of shares held
            dividend_per_share: Dividend amount per share
            withholding_rate: Withholding rate (default 15% for US with W-8BEN)
            currency: Currency of payment
            notes: Optional notes

        Returns:
            Created DividendRecord
        """
        if withholding_rate is None:
            withholding_rate = self.default_withholding_rate

        gross_amount = shares * dividend_per_share
        withholding_amount = gross_amount * withholding_rate
        net_amount = gross_amount - withholding_amount

        record = DividendRecord(
            symbol=symbol,
            payment_date=payment_date,
            shares=shares,
            gross_per_share=dividend_per_share,
            gross_amount=gross_amount,
            withholding_rate=withholding_rate,
            withholding_amount=withholding_amount,
            net_amount=net_amount,
            currency=currency,
            notes=notes,
        )

        self.records.append(record)
        logger.info(
            f"Recorded dividend: {symbol} ${gross_amount:.2f} gross, "
            f"${withholding_amount:.2f} withheld"
        )

        return record

    def record_from_broker(
        self,
        symbol: str,
        payment_date: datetime,
        gross_amount: float,
        net_amount: float,
        currency: str = "USD",
    ) -> DividendRecord:
        """
        Record dividend from broker statement (gross and net known).

        Useful when importing from broker statements where withholding
        is calculated by the broker.

        Args:
            symbol: Ticker symbol
            payment_date: Payment date
            gross_amount: Gross dividend
            net_amount: Net amount received
            currency: Currency

        Returns:
            Created DividendRecord
        """
        withholding_amount = gross_amount - net_amount
        withholding_rate = withholding_amount / gross_amount if gross_amount > 0 else 0

        record = DividendRecord(
            symbol=symbol,
            payment_date=payment_date,
            shares=0,  # Unknown from this import method
            gross_per_share=0,
            gross_amount=gross_amount,
            withholding_rate=withholding_rate,
            withholding_amount=withholding_amount,
            net_amount=net_amount,
            currency=currency,
            notes="Imported from broker statement",
        )

        self.records.append(record)
        return record

    def get_annual_summary(self, year: int) -> AnnualDividendSummary:
        """
        Get annual dividend summary.

        Args:
            year: Tax year

        Returns:
            AnnualDividendSummary
        """
        year_records = [r for r in self.records if r.payment_date.year == year]

        total_gross = sum(r.gross_amount for r in year_records)
        total_withholding = sum(r.withholding_amount for r in year_records)
        total_net = sum(r.net_amount for r in year_records)

        # By symbol
        dividends_by_symbol: Dict[str, float] = {}
        withholding_by_symbol: Dict[str, float] = {}

        for r in year_records:
            dividends_by_symbol[r.symbol] = (
                dividends_by_symbol.get(r.symbol, 0) + r.gross_amount
            )
            withholding_by_symbol[r.symbol] = (
                withholding_by_symbol.get(r.symbol, 0) + r.withholding_amount
            )

        return AnnualDividendSummary(
            year=year,
            total_gross=total_gross,
            total_withholding=total_withholding,
            total_net=total_net,
            n_payments=len(year_records),
            dividends_by_symbol=dividends_by_symbol,
            withholding_by_symbol=withholding_by_symbol,
            records=year_records,
        )

    def get_ytd_summary(self) -> AnnualDividendSummary:
        """Get year-to-date summary."""
        return self.get_annual_summary(datetime.now().year)

    def get_symbol_history(self, symbol: str) -> List[DividendRecord]:
        """Get dividend history for a symbol."""
        return [r for r in self.records if r.symbol == symbol]

    def estimate_annual_dividends(
        self,
        holdings: Dict[str, int],
        dividend_yields: Dict[str, float],
        prices: Dict[str, float],
    ) -> Dict:
        """
        Estimate annual dividends for current holdings.

        Args:
            holdings: Symbol -> shares held
            dividend_yields: Symbol -> annual dividend yield
            prices: Symbol -> current price

        Returns:
            Dict with estimated dividends
        """
        total_gross = 0
        total_withholding = 0
        by_symbol = {}

        for symbol, shares in holdings.items():
            if symbol not in dividend_yields or symbol not in prices:
                continue

            annual_yield = dividend_yields[symbol]
            price = prices[symbol]
            position_value = shares * price

            annual_dividend = position_value * annual_yield
            withholding = annual_dividend * self.default_withholding_rate

            by_symbol[symbol] = {
                "shares": shares,
                "price": price,
                "yield": annual_yield * 100,
                "annual_gross": annual_dividend,
                "annual_withholding": withholding,
                "annual_net": annual_dividend - withholding,
            }

            total_gross += annual_dividend
            total_withholding += withholding

        return {
            "total_gross": total_gross,
            "total_withholding": total_withholding,
            "total_net": total_gross - total_withholding,
            "withholding_rate": self.default_withholding_rate,
            "by_symbol": by_symbol,
        }

    def save(self, filename: str = "dividends.json") -> Path:
        """Save dividend history."""
        filepath = self.data_dir / filename

        data = {
            "saved_at": datetime.now().isoformat(),
            "default_withholding_rate": self.default_withholding_rate,
            "records": [r.to_dict() for r in self.records],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved dividend data: {filepath}")
        return filepath

    def load(self, filename: str = "dividends.json") -> None:
        """Load dividend history."""
        filepath = self.data_dir / filename

        with open(filepath) as f:
            data = json.load(f)

        self.records = [
            DividendRecord.from_dict(r) for r in data.get("records", [])
        ]

        logger.info(f"Loaded dividend data: {len(self.records)} records")

    def print_summary(self, year: int = None) -> None:
        """Print annual summary to console."""
        if year is None:
            year = datetime.now().year

        summary = self.get_annual_summary(year)

        print("\n" + "=" * 60)
        print(f"DIVIDEND INCOME SUMMARY - {year}")
        print("=" * 60)

        print(f"\nPayments: {summary.n_payments}")
        print(f"Gross Dividends: ${summary.total_gross:,.2f}")
        print(f"US Withholding:  ${summary.total_withholding:,.2f}")
        print(f"Net Received:    ${summary.total_net:,.2f}")

        if summary.dividends_by_symbol:
            print("\n--- BY SYMBOL ---")
            print(f"{'Symbol':<10} {'Gross':>12} {'Withheld':>12} {'Net':>12}")
            print("-" * 48)

            for symbol in sorted(summary.dividends_by_symbol.keys()):
                gross = summary.dividends_by_symbol[symbol]
                withheld = summary.withholding_by_symbol.get(symbol, 0)
                net = gross - withheld
                print(f"{symbol:<10} ${gross:>11,.2f} ${withheld:>11,.2f} ${net:>11,.2f}")

        print("\n" + "=" * 60)
        print("Note: Gross dividends are taxable as income in Ireland.")
        print("US withholding tax may be credited against Irish tax due.")
        print("=" * 60)
