"""
Tax report generation for Ireland Revenue.

Generates comprehensive tax reports for annual filing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .ireland_cgt import IrelandCGTCalculator, AnnualCGTSummary
from .dividend_tracker import DividendTracker, AnnualDividendSummary
from ..config import REPORTS_DIR

logger = logging.getLogger(__name__)


class TaxReporter:
    """
    Generates tax reports for Ireland Revenue.

    Combines CGT and dividend information into comprehensive
    annual tax reports.

    Example:
        reporter = TaxReporter(cgt_calc, dividend_tracker)
        report = reporter.generate_annual_report(2024)
        reporter.export_pdf(report, "tax_report_2024.pdf")
    """

    def __init__(
        self,
        cgt_calculator: Optional[IrelandCGTCalculator] = None,
        dividend_tracker: Optional[DividendTracker] = None,
        reports_dir: Optional[Path] = None,
    ):
        """
        Initialize the tax reporter.

        Args:
            cgt_calculator: CGT calculator instance
            dividend_tracker: Dividend tracker instance
            reports_dir: Directory for output reports
        """
        self.cgt = cgt_calculator or IrelandCGTCalculator()
        self.dividends = dividend_tracker or DividendTracker()
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_annual_report(self, year: int) -> Dict:
        """
        Generate comprehensive annual tax report.

        Args:
            year: Tax year

        Returns:
            Dict containing full report data
        """
        cgt_summary = self.cgt.get_annual_summary(year)
        div_summary = self.dividends.get_annual_summary(year)

        report = {
            "year": year,
            "generated_at": datetime.now().isoformat(),
            "capital_gains": self._format_cgt_section(cgt_summary),
            "dividend_income": self._format_dividend_section(div_summary),
            "summary": self._generate_summary(cgt_summary, div_summary),
        }

        return report

    def _format_cgt_section(self, summary: AnnualCGTSummary) -> Dict:
        """Format CGT section of report."""
        gains_detail = []
        for g in summary.gains:
            gains_detail.append({
                "date": g.sale_date.strftime("%Y-%m-%d"),
                "asset": g.symbol,
                "shares": g.shares_sold,
                "proceeds": g.sale_proceeds,
                "cost_basis": g.cost_basis,
                "gain": g.gross_gain,
                "holding_period_days": g.holding_period_days,
            })

        losses_detail = []
        for l in summary.losses:
            losses_detail.append({
                "date": l.sale_date.strftime("%Y-%m-%d"),
                "asset": l.symbol,
                "shares": l.shares_sold,
                "proceeds": l.sale_proceeds,
                "cost_basis": l.cost_basis,
                "loss": abs(l.gross_gain),
                "holding_period_days": l.holding_period_days,
            })

        return {
            "n_disposals": summary.n_disposals,
            "total_proceeds": summary.total_proceeds,
            "total_cost_basis": summary.total_cost_basis,
            "gross_gains": summary.gross_gains,
            "gross_losses": summary.gross_losses,
            "net_gain": summary.net_gain,
            "annual_exemption": summary.annual_exemption,
            "taxable_gain": summary.taxable_gain,
            "cgt_rate": summary.cgt_rate,
            "estimated_tax": summary.estimated_tax,
            "gains_detail": gains_detail,
            "losses_detail": losses_detail,
        }

    def _format_dividend_section(self, summary: AnnualDividendSummary) -> Dict:
        """Format dividend section of report."""
        payments_detail = []
        for r in summary.records:
            payments_detail.append({
                "date": r.payment_date.strftime("%Y-%m-%d"),
                "symbol": r.symbol,
                "gross": r.gross_amount,
                "withholding": r.withholding_amount,
                "withholding_rate": r.withholding_rate,
                "net": r.net_amount,
                "currency": r.currency,
            })

        return {
            "n_payments": summary.n_payments,
            "total_gross": summary.total_gross,
            "total_us_withholding": summary.total_withholding,
            "total_net": summary.total_net,
            "by_symbol": summary.dividends_by_symbol,
            "withholding_by_symbol": summary.withholding_by_symbol,
            "payments_detail": payments_detail,
            "notes": [
                "Gross dividend income is taxable as investment income in Ireland.",
                "US withholding tax may be credited against Irish income tax.",
                "Standard US withholding rate with W-8BEN: 15%",
            ],
        }

    def _generate_summary(
        self, cgt: AnnualCGTSummary, div: AnnualDividendSummary
    ) -> Dict:
        """Generate overall summary."""
        return {
            "cgt_estimated_liability": cgt.estimated_tax,
            "dividend_gross_income": div.total_gross,
            "us_withholding_credit": div.total_withholding,
            "key_dates": {
                "cgt_preliminary_return": f"{cgt.year + 1}-10-31",
                "cgt_final_return": f"{cgt.year + 1}-11-15",
                "income_tax_return": f"{cgt.year + 1}-10-31",
            },
            "notes": [
                "CGT is due in two installments:",
                f"  - Jan 1 - Sep 30 disposals: pay by Dec 15, {cgt.year}",
                f"  - Oct 1 - Dec 31 disposals: pay by Jan 31, {cgt.year + 1}",
                "Dividend income should be included on Form 11/Form 12.",
            ],
        }

    def export_json(self, report: Dict, filename: Optional[str] = None) -> Path:
        """
        Export report as JSON.

        Args:
            report: Report data
            filename: Output filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"tax_report_{report['year']}.json"

        filepath = self.reports_dir / filename

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported tax report: {filepath}")
        return filepath

    def export_csv(self, report: Dict) -> Dict[str, Path]:
        """
        Export report as multiple CSV files.

        Args:
            report: Report data

        Returns:
            Dict of section name -> filepath
        """
        import csv

        year = report["year"]
        paths = {}

        # CGT gains
        cgt_gains_path = self.reports_dir / f"cgt_gains_{year}.csv"
        with open(cgt_gains_path, "w", newline="") as f:
            if report["capital_gains"]["gains_detail"]:
                writer = csv.DictWriter(
                    f, fieldnames=report["capital_gains"]["gains_detail"][0].keys()
                )
                writer.writeheader()
                writer.writerows(report["capital_gains"]["gains_detail"])
        paths["cgt_gains"] = cgt_gains_path

        # CGT losses
        cgt_losses_path = self.reports_dir / f"cgt_losses_{year}.csv"
        with open(cgt_losses_path, "w", newline="") as f:
            if report["capital_gains"]["losses_detail"]:
                writer = csv.DictWriter(
                    f, fieldnames=report["capital_gains"]["losses_detail"][0].keys()
                )
                writer.writeheader()
                writer.writerows(report["capital_gains"]["losses_detail"])
        paths["cgt_losses"] = cgt_losses_path

        # Dividends
        div_path = self.reports_dir / f"dividends_{year}.csv"
        with open(div_path, "w", newline="") as f:
            if report["dividend_income"]["payments_detail"]:
                writer = csv.DictWriter(
                    f, fieldnames=report["dividend_income"]["payments_detail"][0].keys()
                )
                writer.writeheader()
                writer.writerows(report["dividend_income"]["payments_detail"])
        paths["dividends"] = div_path

        logger.info(f"Exported CSV reports: {list(paths.values())}")
        return paths

    def print_report(self, year: int = None) -> None:
        """Print annual tax report to console."""
        if year is None:
            year = datetime.now().year

        report = self.generate_annual_report(year)

        print("\n" + "=" * 70)
        print(f"IRELAND TAX REPORT - {year}")
        print("=" * 70)

        # CGT Section
        cgt = report["capital_gains"]
        print("\n" + "-" * 70)
        print("CAPITAL GAINS TAX")
        print("-" * 70)
        print(f"Disposals:        {cgt['n_disposals']}")
        print(f"Total Proceeds:   EUR {cgt['total_proceeds']:,.2f}")
        print(f"Total Cost Basis: EUR {cgt['total_cost_basis']:,.2f}")
        print(f"Gross Gains:      EUR {cgt['gross_gains']:,.2f}")
        print(f"Gross Losses:     EUR {cgt['gross_losses']:,.2f}")
        print(f"Net Gain:         EUR {cgt['net_gain']:,.2f}")
        print(f"Annual Exemption: EUR {cgt['annual_exemption']:,.2f}")
        print(f"Taxable Gain:     EUR {cgt['taxable_gain']:,.2f}")
        print(f"CGT Rate:         {cgt['cgt_rate'] * 100:.0f}%")
        print(f"ESTIMATED CGT:    EUR {cgt['estimated_tax']:,.2f}")

        # Dividend Section
        div = report["dividend_income"]
        print("\n" + "-" * 70)
        print("DIVIDEND INCOME")
        print("-" * 70)
        print(f"Payments:          {div['n_payments']}")
        print(f"Gross Dividends:   USD {div['total_gross']:,.2f}")
        print(f"US Withholding:    USD {div['total_us_withholding']:,.2f}")
        print(f"Net Received:      USD {div['total_net']:,.2f}")
        print("\nNote: Dividends taxable as income; US withholding may be credited.")

        # Summary
        summary = report["summary"]
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"Estimated CGT Liability:    EUR {summary['cgt_estimated_liability']:,.2f}")
        print(f"Gross Dividend Income:      USD {summary['dividend_gross_income']:,.2f}")
        print(f"US Withholding Tax Credit:  USD {summary['us_withholding_credit']:,.2f}")

        print("\nKey Dates:")
        for date_type, date_val in summary["key_dates"].items():
            print(f"  {date_type.replace('_', ' ').title()}: {date_val}")

        print("\n" + "=" * 70)
