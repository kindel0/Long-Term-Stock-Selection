"""
EODHD data downloader.

Downloads bulk financial data from EODHD API and saves as Parquet files.
Supports prices, fundamentals, company info, and macro data.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from ..config import (
    EODHD_DIR,
    EODHD_CONFIG,
    EODHD_FILES,
    EODHD_FUNDAMENTAL_FIELDS,
    EODHD_MACRO_INDICATORS,
)
from .column_mapping import (
    EODHD_PRICE_MAPPING,
    EODHD_INCOME_MAPPING,
    EODHD_BALANCE_MAPPING,
    EODHD_CASHFLOW_MAPPING,
    EODHD_COMPANY_MAPPING,
    is_common_stock,
    map_columns,
)

logger = logging.getLogger(__name__)


class EODHDDownloader:
    """
    Downloads and caches EODHD data as Parquet files.

    Supports resumable downloads - interrupted downloads can be continued
    from where they left off.

    Respects EODHD API limits:
    - Daily limit: 100,000 API calls (configurable)
    - Fundamental requests cost 10 API calls each
    - Price/EOD requests cost 1 API call each
    - Exchange symbol list costs 1 API call

    Usage:
        downloader = EODHDDownloader(api_key="your_key")
        downloader.download_all(output_dir="data/eodhd")

        # Resume interrupted download
        downloader.download_all(output_dir="data/eodhd", resume=True)
    """

    # API call costs per request type (from EODHD documentation)
    API_CALL_COSTS = {
        "fundamentals": 10,  # Fundamental API costs 10 calls per request
        "eod": 1,            # EOD/price data costs 1 call per request
        "exchange_list": 1,  # Exchange symbol list costs 1 call
        "bulk_eod": 1,       # Bulk EOD costs 1 call
        "macro": 1,          # Macro data costs 1 call
    }

    DEFAULT_DAILY_LIMIT = 100_000  # EODHD default daily limit

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
        daily_limit: Optional[int] = None,
    ):
        """
        Initialize the downloader.

        Args:
            api_key: EODHD API key (or set EODHD_API_KEY env var)
            output_dir: Directory to save Parquet files
            daily_limit: Override daily API call limit (default: 100,000)
        """
        self.api_key = api_key or os.environ.get("EODHD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EODHD API key required. Set EODHD_API_KEY env var or pass api_key."
            )

        self.output_dir = Path(output_dir) if output_dir else EODHD_DIR
        self.base_url = EODHD_CONFIG["base_url"]
        self.exchange = EODHD_CONFIG["exchange"]
        self.daily_limit = daily_limit or self.DEFAULT_DAILY_LIMIT

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 60 / EODHD_CONFIG["requests_per_minute"]

        # Track API calls (weighted by cost)
        self.api_calls = 0  # Total weighted API calls
        self.api_requests = 0  # Total requests made

        # Progress tracking for resume
        self._progress_file = self.output_dir / ".download_progress.json"

    def get_remaining_quota(self) -> int:
        """Get remaining API calls for today."""
        return max(0, self.daily_limit - self.api_calls)

    def check_quota(self, request_type: str, count: int = 1) -> bool:
        """
        Check if we have enough quota for the planned requests.

        Args:
            request_type: Type of request (fundamentals, eod, etc.)
            count: Number of requests planned

        Returns:
            True if quota is sufficient, False otherwise
        """
        cost_per_request = self.API_CALL_COSTS.get(request_type, 1)
        total_cost = cost_per_request * count
        return self.api_calls + total_cost <= self.daily_limit

    def estimate_required_calls(self, n_symbols: int, include_prices: bool, include_fundamentals: bool) -> int:
        """Estimate total API calls needed for a download."""
        total = 1  # Exchange symbol list
        if include_prices:
            total += n_symbols * self.API_CALL_COSTS["eod"]
        if include_fundamentals:
            total += n_symbols * self.API_CALL_COSTS["fundamentals"]
        return total

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _load_progress(self) -> Dict:
        """Load download progress from disk."""
        if self._progress_file.exists():
            with open(self._progress_file) as f:
                return json.load(f)
        return {
            "completed_symbols": {"prices": [], "fundamentals": []},
            "completed_dates": [],
            "stage": None,
        }

    def _save_progress(self, progress: Dict):
        """Save download progress to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self._progress_file, "w") as f:
            json.dump(progress, f)

    def _clear_progress(self):
        """Clear progress file after successful completion."""
        if self._progress_file.exists():
            self._progress_file.unlink()

    def _is_stage_complete(self, stage: str) -> bool:
        """Check if a download stage is already complete (final files exist)."""
        if stage == "prices":
            return (self.output_dir / EODHD_FILES["prices"]).exists()
        elif stage == "fundamentals":
            # Check if any of the fundamental files exist
            return (
                (self.output_dir / EODHD_FILES["income"]).exists() or
                (self.output_dir / EODHD_FILES["balance"]).exists() or
                (self.output_dir / EODHD_FILES["cashflow"]).exists()
            )
        elif stage == "macro":
            return (self.output_dir / EODHD_FILES["macro"]).exists()
        elif stage == "companies":
            return (self.output_dir / EODHD_FILES["companies"]).exists()
        return False

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: int = 60,
        request_type: str = "eod",
    ) -> Any:
        """
        Make an API request with rate limiting and quota tracking.

        Args:
            endpoint: API endpoint
            params: Query parameters
            timeout: Request timeout in seconds
            request_type: Type of request for API call cost calculation

        Returns:
            JSON response data

        Raises:
            ValueError: If daily API call limit would be exceeded
        """
        # Check quota before making request
        cost = self.API_CALL_COSTS.get(request_type, 1)
        if self.api_calls + cost > self.daily_limit:
            raise ValueError(
                f"Daily API call limit ({self.daily_limit:,}) would be exceeded. "
                f"Current usage: {self.api_calls:,}, request cost: {cost}. "
                f"Wait for limit reset at midnight GMT or increase your daily limit at https://eodhd.com"
            )

        self._rate_limit()

        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params["api_token"] = self.api_key
        params["fmt"] = "json"

        logger.debug(f"Request: {endpoint} (cost: {cost} API calls)")
        response = requests.get(url, params=params, timeout=timeout)
        self.api_calls += cost
        self.api_requests += 1

        if response.status_code == 402:
            raise ValueError(
                f"API limit exceeded (402 Payment Required). "
                f"Requests made: {self.api_requests}, API calls used: {self.api_calls:,}. "
                f"Wait for daily limit reset at midnight GMT."
            )

        if response.status_code != 200:
            logger.error(f"API error {response.status_code}: {response.text[:200]}")
            response.raise_for_status()

        return response.json()

    def get_exchange_symbols(
        self,
        exchange: Optional[str] = None,
        security_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get list of symbols for an exchange.

        Args:
            exchange: Exchange code (default: US)
            security_type: Filter by type (e.g., "Common Stock")

        Returns:
            DataFrame with symbol information
        """
        exchange = exchange or self.exchange
        endpoint = f"exchange-symbol-list/{exchange}"

        data = self._make_request(endpoint, request_type="exchange_list")
        df = pd.DataFrame(data)

        logger.info(f"Got {len(df)} symbols from {exchange}")

        # Filter by security type
        if security_type and "Type" in df.columns:
            df = df[df["Type"].str.lower().str.contains(security_type.lower(), na=False)]
            logger.info(f"Filtered to {len(df)} {security_type} symbols")

        return df

    def download_eod_bulk(
        self,
        date: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download bulk end-of-day prices.

        Args:
            date: Date string YYYY-MM-DD (default: latest)
            exchange: Exchange code

        Returns:
            DataFrame with EOD prices
        """
        exchange = exchange or self.exchange
        endpoint = f"eod-bulk-last-day/{exchange}"

        params = {}
        if date:
            params["date"] = date

        data = self._make_request(endpoint, params)
        df = pd.DataFrame(data)

        logger.info(f"Downloaded {len(df)} EOD price records")
        return df

    def download_historical_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download historical prices for a single symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD

        Returns:
            DataFrame with historical prices
        """
        endpoint = f"eod/{symbol}.{self.exchange}"

        params = {}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        data = self._make_request(endpoint, params)
        df = pd.DataFrame(data)
        df["code"] = symbol

        return df

    def download_bulk_eod(
        self,
        date: str,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Download bulk EOD prices for a specific date.

        This endpoint is typically available on lower subscription tiers.

        Args:
            date: Date string YYYY-MM-DD
            symbols: Optional list of symbols to filter (default: all)

        Returns:
            DataFrame with EOD prices for that date
        """
        endpoint = f"eod-bulk-last-day/{self.exchange}"
        params = {"date": date}

        if symbols:
            params["symbols"] = ",".join(symbols[:500])  # API limit

        data = self._make_request(endpoint, params, timeout=120, request_type="bulk_eod")
        df = pd.DataFrame(data)

        return df

    def download_bulk_historical(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        method: str = "auto",
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical prices using the most efficient available method.

        Args:
            symbols: List of stock symbols
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD (default: today)
            method: 'bulk' (date-by-date), 'individual' (symbol-by-symbol), or 'auto'
            resume: Whether to resume from previous progress

        Returns:
            Combined DataFrame with all prices
        """
        import datetime

        end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")

        if method == "auto":
            # Test which method works
            method = self._detect_available_method(symbols[0], start_date)
            logger.info(f"Using {method} download method")

        if method == "bulk":
            return self._download_via_bulk_dates(symbols, start_date, end_date, resume=resume)
        else:
            return self._download_via_individual(symbols, start_date, end_date, resume=resume)

    def _detect_available_method(self, test_symbol: str, start_date: str) -> str:
        """Detect which download method is available for this API key."""
        # Try individual endpoint first (preferred for historical)
        try:
            endpoint = f"eod/{test_symbol}.{self.exchange}"
            params = {"from": start_date, "to": start_date}
            self._make_request(endpoint, params, request_type="eod")
            return "individual"
        except requests.HTTPError as e:
            if e.response.status_code == 403:
                logger.info("Individual EOD endpoint not available, trying bulk")
            else:
                raise
        except ValueError as e:
            if "limit" in str(e).lower():
                raise
            raise

        # Try bulk endpoint
        try:
            import datetime
            test_date = datetime.date.today() - datetime.timedelta(days=7)
            self.download_bulk_eod(test_date.strftime("%Y-%m-%d"))
            return "bulk"
        except requests.HTTPError as e:
            if e.response.status_code == 403:
                raise ValueError(
                    "Neither individual nor bulk EOD endpoints available. "
                    "Please check your EODHD subscription tier."
                )
            raise

    def _download_via_bulk_dates(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        resume: bool = True,
        save_interval: int = 24,
    ) -> pd.DataFrame:
        """Download historical data by iterating through dates using bulk endpoint."""
        import datetime

        start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        symbols_set = set(symbols)

        # Load existing progress
        progress = self._load_progress() if resume else {"completed_dates": []}
        completed_dates = set(progress.get("completed_dates", []))

        # Load existing data if resuming
        prices_partial_path = self.output_dir / "prices_partial.parquet"
        if resume and prices_partial_path.exists():
            existing_df = pd.read_parquet(prices_partial_path)
            all_data = [existing_df]
            logger.info(f"Resuming price download: {len(completed_dates)} dates already done")
        else:
            all_data = []

        # Generate month-end dates to sample (reduces API calls significantly)
        current = start.replace(day=28)  # Use 28 to avoid month-end issues
        dates_to_fetch = []

        while current <= end:
            # Get last trading day of month (approximate with last calendar day)
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)
            last_day = next_month - datetime.timedelta(days=1)

            if last_day >= start and last_day <= end:
                date_str = last_day.strftime("%Y-%m-%d")
                if date_str not in completed_dates:
                    dates_to_fetch.append(last_day)

            # Move to next month
            current = next_month

        if not dates_to_fetch:
            logger.info("All price dates already downloaded")
            if prices_partial_path.exists():
                return pd.read_parquet(prices_partial_path)
            return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

        logger.info(f"Fetching {len(dates_to_fetch)} month-end dates via bulk endpoint ({len(completed_dates)} already done)")

        for i, date in enumerate(dates_to_fetch):
            if (i + 1) % 12 == 0:
                logger.info(f"  Progress: {i + 1}/{len(dates_to_fetch)} dates")

            try:
                df = self.download_bulk_eod(date.strftime("%Y-%m-%d"))
                if not df.empty and "code" in df.columns:
                    # Filter to our symbols
                    df = df[df["code"].isin(symbols_set)]
                    all_data.append(df)
                    completed_dates.add(date.strftime("%Y-%m-%d"))
            except Exception as e:
                logger.warning(f"Failed to get bulk data for {date}: {e}")
                continue

            # Save progress periodically
            if (i + 1) % save_interval == 0:
                self._save_incremental_prices(all_data, completed_dates, progress)

        # Final save
        self._save_incremental_prices(all_data, completed_dates, progress)

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def _download_via_individual(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        resume: bool = True,
        save_interval: int = 100,
    ) -> pd.DataFrame:
        """Download historical data symbol by symbol with resume support."""
        # Load existing progress
        progress = self._load_progress() if resume else {"completed_symbols": {"prices": []}}
        completed = set(progress.get("completed_symbols", {}).get("prices", []))

        # Load existing data if resuming
        prices_partial_path = self.output_dir / "prices_partial.parquet"
        if resume and prices_partial_path.exists():
            existing_df = pd.read_parquet(prices_partial_path)
            all_prices = [existing_df]
            logger.info(f"Resuming price download: {len(completed)} symbols already done")
        else:
            all_prices = []

        # Filter to remaining symbols
        remaining = [s for s in symbols if s not in completed]
        if not remaining:
            logger.info("All price symbols already downloaded")
            if prices_partial_path.exists():
                return pd.read_parquet(prices_partial_path)
            return pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()

        initial_completed = len(completed)
        logger.info(f"Downloading prices for {len(remaining)} symbols ({initial_completed} already done)")

        for i, symbol in enumerate(remaining):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{len(remaining)} ({initial_completed + i + 1}/{len(symbols)} total)")

            try:
                df = self.download_historical_prices(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                all_prices.append(df)
                completed.add(symbol)
            except Exception as e:
                logger.warning(f"Failed to get prices for {symbol}: {e}")
                continue

            # Save progress periodically
            if (i + 1) % save_interval == 0:
                self._save_incremental_prices_individual(all_prices, completed, progress)

        # Final save
        self._save_incremental_prices_individual(all_prices, completed, progress)

        if not all_prices:
            return pd.DataFrame()

        return pd.concat(all_prices, ignore_index=True)

    def _save_incremental_prices(
        self,
        data_frames: List[pd.DataFrame],
        completed_dates: set,
        progress: Dict,
    ):
        """Save incremental price progress (bulk dates method)."""
        if data_frames:
            df = pd.concat(data_frames, ignore_index=True)
            partial_path = self.output_dir / "prices_partial.parquet"
            df.to_parquet(partial_path, index=False)

        progress["completed_dates"] = list(completed_dates)
        self._save_progress(progress)
        logger.debug(f"Saved progress: {len(completed_dates)} price dates")

    def _save_incremental_prices_individual(
        self,
        data_frames: List[pd.DataFrame],
        completed: set,
        progress: Dict,
    ):
        """Save incremental price progress (individual symbols method)."""
        if data_frames:
            df = pd.concat(data_frames, ignore_index=True)
            partial_path = self.output_dir / "prices_partial.parquet"
            df.to_parquet(partial_path, index=False)

        progress["completed_symbols"]["prices"] = list(completed)
        self._save_progress(progress)
        logger.debug(f"Saved progress: {len(completed)} price symbols")

    def download_fundamentals(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Download fundamental data for a symbol.

        Note: Each fundamentals request costs 10 API calls.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamental data
        """
        endpoint = f"fundamentals/{symbol}.{self.exchange}"
        return self._make_request(endpoint, request_type="fundamentals")

    def download_bulk_fundamentals(
        self,
        symbols: List[str],
        batch_size: int = 100,
        resume: bool = True,
        save_interval: int = 500,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download fundamentals for multiple symbols with resume support.

        Also extracts company info (sector, industry, GICS codes) for filtering.

        Args:
            symbols: List of stock symbols
            batch_size: Symbols per batch for logging
            resume: Whether to resume from previous progress
            save_interval: Save progress every N symbols

        Returns:
            Tuple of (fundamentals DataFrame, company info DataFrame)
        """
        # Load existing progress
        progress = self._load_progress() if resume else {"completed_symbols": {"fundamentals": []}}
        completed = set(progress.get("completed_symbols", {}).get("fundamentals", []))

        # Load existing data if resuming
        fundamentals_path = self.output_dir / "fundamentals_partial.parquet"
        companies_path = self.output_dir / "companies_detail_partial.parquet"

        if resume and fundamentals_path.exists():
            existing_df = pd.read_parquet(fundamentals_path)
            all_records = existing_df.to_dict("records")
            logger.info(f"Resuming fundamentals download: {len(completed)} symbols already done")
        else:
            all_records = []

        if resume and companies_path.exists():
            existing_companies = pd.read_parquet(companies_path)
            all_company_info = existing_companies.to_dict("records")
        else:
            all_company_info = []

        # Filter to remaining symbols
        remaining = [s for s in symbols if s not in completed]
        if not remaining:
            logger.info("All fundamentals already downloaded")
            fundamentals_df = pd.read_parquet(fundamentals_path) if fundamentals_path.exists() else pd.DataFrame(all_records)
            companies_df = pd.read_parquet(companies_path) if companies_path.exists() else pd.DataFrame(all_company_info)
            return fundamentals_df, companies_df

        initial_completed = len(completed)
        logger.info(f"Downloading fundamentals for {len(remaining)} symbols ({initial_completed} already done)")
        logger.info(f"API calls remaining: {self.get_remaining_quota():,} (each fundamental = 10 calls)")

        for i, symbol in enumerate(remaining):
            if (i + 1) % 100 == 0:
                logger.info(
                    f"  Progress: {i + 1}/{len(remaining)} ({initial_completed + i + 1}/{len(symbols)} total) "
                    f"| API calls used: {self.api_calls:,}"
                )

            try:
                data = self.download_fundamentals(symbol)
                records, company_info = self._extract_quarterly_data(symbol, data)
                all_records.extend(records)
                if company_info:
                    all_company_info.append(company_info)
                completed.add(symbol)
            except ValueError as e:
                # API limit reached
                logger.error(f"API limit reached: {e}")
                logger.info("Saving progress. Run again with --resume to continue tomorrow.")
                break
            except Exception as e:
                logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
                continue

            # Save progress periodically
            if (i + 1) % save_interval == 0:
                self._save_incremental_fundamentals(all_records, all_company_info, completed, progress)

        # Final save
        self._save_incremental_fundamentals(all_records, all_company_info, completed, progress)

        fundamentals_df = pd.DataFrame(all_records)
        companies_df = pd.DataFrame(all_company_info)

        logger.info(f"Downloaded {len(fundamentals_df)} quarterly fundamental records")
        logger.info(f"Extracted company info for {len(companies_df)} companies")

        return fundamentals_df, companies_df

    def _save_incremental_fundamentals(
        self,
        records: List[Dict],
        company_info: List[Dict],
        completed: set,
        progress: Dict,
    ):
        """Save incremental fundamentals and company info progress."""
        if records:
            df = pd.DataFrame(records)
            partial_path = self.output_dir / "fundamentals_partial.parquet"
            df.to_parquet(partial_path, index=False)

        if company_info:
            companies_df = pd.DataFrame(company_info)
            companies_path = self.output_dir / "companies_detail_partial.parquet"
            companies_df.to_parquet(companies_path, index=False)

        progress["completed_symbols"]["fundamentals"] = list(completed)
        self._save_progress(progress)
        logger.debug(f"Saved progress: {len(completed)} fundamentals symbols")

    def _extract_quarterly_data(
        self,
        symbol: str,
        data: Dict[str, Any],
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """
        Extract quarterly financial data AND company info from fundamentals response.

        Args:
            symbol: Stock symbol
            data: Raw fundamentals data

        Returns:
            Tuple of (list of quarterly records, company info dict)
        """
        records = []

        # Extract company general info (sector, industry, etc.)
        general = data.get("General", {})
        company_info = None
        if general:
            company_info = {
                "code": symbol,
                "name": general.get("Name"),
                "exchange": general.get("Exchange"),
                "currency": general.get("CurrencyCode"),
                "country": general.get("CountryName"),
                "sector": general.get("Sector"),
                "industry": general.get("Industry"),
                "gics_sector": general.get("GicSector"),
                "gics_group": general.get("GicGroup"),
                "gics_industry": general.get("GicIndustry"),
                "gics_sub_industry": general.get("GicSubIndustry"),
                "description": general.get("Description"),
                "full_time_employees": general.get("FullTimeEmployees"),
                "ipo_date": general.get("IPODate"),
                "is_delisted": general.get("IsDelisted"),
            }

        # Get quarterly data from financials section
        financials = data.get("Financials", {})

        # Income statement quarterly
        income_q = financials.get("Income_Statement", {}).get("quarterly", {})
        balance_q = financials.get("Balance_Sheet", {}).get("quarterly", {})
        cashflow_q = financials.get("Cash_Flow", {}).get("quarterly", {})

        # Get all dates from all statements
        all_dates = set()
        all_dates.update(income_q.keys())
        all_dates.update(balance_q.keys())
        all_dates.update(cashflow_q.keys())

        for date_key in sorted(all_dates):
            record = {"code": symbol, "date": date_key}

            # Add company sector/industry to each record for easy filtering
            if company_info:
                record["sector"] = company_info.get("sector")
                record["industry"] = company_info.get("industry")
                record["gics_sector"] = company_info.get("gics_sector")

            # Add income statement items - ALL of them
            if date_key in income_q:
                income_data = income_q[date_key]
                record.update(self._flatten_statement(income_data, "income"))
                # Get filing date if available
                if "filing_date" in income_data:
                    record["filing_date"] = income_data["filing_date"]

            # Add balance sheet items - ALL of them
            if date_key in balance_q:
                balance_data = balance_q[date_key]
                record.update(self._flatten_statement(balance_data, "balance"))

            # Add cash flow items - ALL of them
            if date_key in cashflow_q:
                cashflow_data = cashflow_q[date_key]
                record.update(self._flatten_statement(cashflow_data, "cashflow"))

            records.append(record)

        return records, company_info

    def _flatten_statement(
        self,
        data: Dict[str, Any],
        statement_type: str,
    ) -> Dict[str, Any]:
        """
        Flatten a financial statement dictionary.

        Args:
            data: Statement data
            statement_type: Type of statement

        Returns:
            Flattened dictionary
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, (int, float, str)) and key not in ["date", "filing_date"]:
                # Convert to numeric if possible
                try:
                    result[key] = float(value) if value is not None else None
                except (ValueError, TypeError):
                    result[key] = value
        return result

    def download_macro_data(
        self,
        country: str = "USA",
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Download macroeconomic data.

        Args:
            country: Country code
            indicators: List of indicator names

        Returns:
            DataFrame with macro data
        """
        indicators = indicators or EODHD_MACRO_INDICATORS
        endpoint = f"macro-indicator/{country}"

        all_data = []
        for indicator in indicators:
            try:
                params = {"indicator": indicator}
                data = self._make_request(endpoint, params, request_type="macro")
                df = pd.DataFrame(data)
                df["indicator"] = indicator
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to get macro indicator {indicator}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"Downloaded {len(result)} macro data points")
        return result

    def download_all(
        self,
        include_prices: bool = True,
        include_fundamentals: bool = True,
        include_companies: bool = True,
        include_macro: bool = False,
        common_stocks_only: bool = True,
        max_symbols: Optional[int] = None,
        skip_on_403: bool = True,
        resume: bool = True,
    ) -> Dict[str, Path]:
        """
        Download all data and save as Parquet files.

        Supports resumable downloads - if interrupted, run again with resume=True
        to continue from where it left off.

        Args:
            include_prices: Download historical prices
            include_fundamentals: Download fundamentals
            include_companies: Download company info
            include_macro: Download macro indicators
            common_stocks_only: Filter to common stocks only
            max_symbols: Limit number of symbols (for testing)
            skip_on_403: Skip data types that return 403 (subscription limits)
            resume: Resume from previous progress if interrupted

        Returns:
            Dictionary of data type -> file path
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        errors = []

        logger.info("=" * 60)
        logger.info("EODHD DATA DOWNLOAD")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Daily API call limit: {self.daily_limit:,}")

        # Get symbol list
        logger.info("\nGetting exchange symbols...")
        try:
            symbols_df = self.get_exchange_symbols()
        except requests.HTTPError as e:
            if e.response.status_code == 403:
                logger.error("API key doesn't have access to exchange symbols endpoint")
                raise ValueError(
                    "Your EODHD subscription doesn't include access to required endpoints. "
                    "Please check your subscription tier at https://eodhd.com"
                )
            raise
        except ValueError as e:
            # API limit exceeded
            raise

        if include_companies:
            companies_path = self.output_dir / EODHD_FILES["companies"]
            symbols_df.to_parquet(companies_path, index=False)
            outputs["companies"] = companies_path
            logger.info(f"Saved {len(symbols_df)} companies to {companies_path}")

        # Filter to common stocks
        if common_stocks_only and "Type" in symbols_df.columns:
            symbols_df = symbols_df[symbols_df["Type"].apply(is_common_stock)]
            logger.info(f"Filtered to {len(symbols_df)} common stocks")

        symbols = list(symbols_df["Code"])
        if max_symbols:
            symbols = symbols[:max_symbols]
            logger.info(f"Limited to {max_symbols} symbols for testing")

        # Estimate API calls and check against limit
        estimated_calls = self.estimate_required_calls(
            len(symbols),
            include_prices=include_prices and not (self._is_stage_complete("prices") and resume),
            include_fundamentals=include_fundamentals and not (self._is_stage_complete("fundamentals") and resume),
        )
        logger.info(f"\nEstimated API calls needed: {estimated_calls:,}")
        logger.info(f"  - Prices: {len(symbols):,} symbols × 1 call = {len(symbols):,} calls")
        logger.info(f"  - Fundamentals: {len(symbols):,} symbols × 10 calls = {len(symbols) * 10:,} calls")
        logger.info(f"Remaining quota: {self.get_remaining_quota():,} calls")

        if estimated_calls > self.daily_limit:
            logger.warning(
                f"\n⚠️  WARNING: Estimated API calls ({estimated_calls:,}) exceed daily limit ({self.daily_limit:,})!"
            )
            logger.warning(
                "The download will stop when the limit is reached. "
                "Use --resume to continue tomorrow after the limit resets."
            )
            logger.warning(
                "Consider downloading in stages: --no-fundamentals first, then --no-prices later."
            )

        # Download prices
        if include_prices:
            if self._is_stage_complete("prices") and resume:
                prices_path = self.output_dir / EODHD_FILES["prices"]
                outputs["prices"] = prices_path
                logger.info(f"\nPrices already downloaded: {prices_path}")
            else:
                logger.info(f"\nDownloading historical prices for {len(symbols)} symbols...")
                try:
                    prices_df = self._download_all_prices(symbols, resume=resume)
                    if not prices_df.empty:
                        prices_path = self.output_dir / EODHD_FILES["prices"]
                        prices_df.to_parquet(prices_path, index=False)
                        outputs["prices"] = prices_path
                        logger.info(f"Saved {len(prices_df)} price records to {prices_path}")
                        # Clean up partial file
                        partial_path = self.output_dir / "prices_partial.parquet"
                        if partial_path.exists():
                            partial_path.unlink()
                    else:
                        logger.warning("No price data retrieved")
                except ValueError as e:
                    if "subscription" in str(e).lower() and skip_on_403:
                        logger.warning(f"Skipping prices: {e}")
                        errors.append(("prices", str(e)))
                    else:
                        raise
                except requests.HTTPError as e:
                    if e.response.status_code == 403 and skip_on_403:
                        logger.warning(f"Skipping prices (403 Forbidden): Your subscription may not include historical EOD data")
                        errors.append(("prices", "403 Forbidden - subscription limit"))
                    else:
                        raise

        # Download fundamentals
        if include_fundamentals:
            if self._is_stage_complete("fundamentals") and resume:
                logger.info(f"\nFundamentals already downloaded")
                # Add existing files to outputs
                for name in ["income", "balance", "cashflow", "fundamentals_all"]:
                    if name == "fundamentals_all":
                        path = self.output_dir / "fundamentals_all.parquet"
                    else:
                        path = self.output_dir / EODHD_FILES[name]
                    if path.exists():
                        outputs[name] = path
                # Check for company detail file
                companies_detail_path = self.output_dir / "companies_detail.parquet"
                if companies_detail_path.exists():
                    outputs["companies_detail"] = companies_detail_path
            else:
                logger.info(f"\nDownloading fundamentals for {len(symbols)} symbols...")
                try:
                    fundamentals_df, companies_df = self.download_bulk_fundamentals(symbols, resume=resume)

                    if not fundamentals_df.empty:
                        # Save ALL fundamentals data (no filtering!)
                        self._save_fundamentals(fundamentals_df, outputs)
                        # Clean up partial file
                        partial_path = self.output_dir / "fundamentals_partial.parquet"
                        if partial_path.exists():
                            partial_path.unlink()
                    else:
                        logger.warning("No fundamental data retrieved")

                    # Save company detail info (sector, industry, GICS codes)
                    if not companies_df.empty:
                        companies_detail_path = self.output_dir / "companies_detail.parquet"
                        companies_df.to_parquet(companies_detail_path, index=False)
                        outputs["companies_detail"] = companies_detail_path
                        logger.info(f"Saved company details (sector/industry) for {len(companies_df)} companies")
                        # Clean up partial
                        partial_path = self.output_dir / "companies_detail_partial.parquet"
                        if partial_path.exists():
                            partial_path.unlink()

                except ValueError as e:
                    # API limit reached - progress already saved
                    logger.error(f"Download stopped: {e}")
                    errors.append(("fundamentals", str(e)))
                except requests.HTTPError as e:
                    if e.response.status_code == 403 and skip_on_403:
                        logger.warning(f"Skipping fundamentals (403 Forbidden): Your subscription may not include fundamental data")
                        errors.append(("fundamentals", "403 Forbidden - subscription limit"))
                    else:
                        raise

        # Download macro
        if include_macro:
            if self._is_stage_complete("macro") and resume:
                macro_path = self.output_dir / EODHD_FILES["macro"]
                outputs["macro"] = macro_path
                logger.info(f"\nMacro data already downloaded: {macro_path}")
            else:
                logger.info("\nDownloading macro data...")
                try:
                    macro_df = self.download_macro_data()
                    if not macro_df.empty:
                        macro_path = self.output_dir / EODHD_FILES["macro"]
                        macro_df.to_parquet(macro_path, index=False)
                        outputs["macro"] = macro_path
                        logger.info(f"Saved {len(macro_df)} macro records to {macro_path}")
                except requests.HTTPError as e:
                    if e.response.status_code == 403 and skip_on_403:
                        logger.warning(f"Skipping macro (403 Forbidden): Your subscription may not include macro data")
                        errors.append(("macro", "403 Forbidden - subscription limit"))
                    else:
                        raise

        # Save metadata
        metadata = {
            "download_timestamp": datetime.now().isoformat(),
            "api_calls": self.api_calls,  # Weighted API calls (fundamentals = 10x)
            "api_requests": self.api_requests,  # Raw request count
            "daily_limit": self.daily_limit,
            "symbols_count": len(symbols),
            "common_stocks_only": common_stocks_only,
            "files": {k: str(v) for k, v in outputs.items()},
            "errors": errors,
        }
        metadata_path = self.output_dir / EODHD_FILES["metadata"]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        outputs["metadata"] = metadata_path

        # Clean up progress file on successful completion
        if not errors:
            self._clear_progress()

        logger.info("\n" + "=" * 60)
        logger.info(f"Download complete.")
        logger.info(f"  Requests made: {self.api_requests:,}")
        logger.info(f"  API calls used: {self.api_calls:,} (weighted)")
        logger.info(f"  Remaining quota: {self.get_remaining_quota():,}")
        if errors:
            logger.warning(f"Some data types were skipped due to subscription limits:")
            for data_type, error in errors:
                logger.warning(f"  - {data_type}: {error}")
        logger.info("=" * 60)

        return outputs

    def _download_all_prices(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical prices for all symbols.

        Automatically detects the best available method based on API subscription.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            resume: Whether to resume from previous progress

        Returns:
            Combined DataFrame with all prices
        """
        start_date = start_date or EODHD_CONFIG["start_date"]

        return self.download_bulk_historical(
            symbols=symbols,
            start_date=start_date,
            method="auto",
            resume=resume,
        )

    def _save_fundamentals(
        self,
        fundamentals_df: pd.DataFrame,
        outputs: Dict[str, Path],
    ):
        """
        Save ALL fundamentals data - don't discard any columns.

        We save everything and filter only at panel build time.
        This prevents data loss and avoids needing to re-download.

        Args:
            fundamentals_df: Combined fundamentals DataFrame
            outputs: Dictionary to update with output paths
        """
        if fundamentals_df.empty:
            logger.warning("No fundamentals data to save")
            return

        # Log what we're saving
        logger.info(f"Saving {len(fundamentals_df)} fundamental records with {len(fundamentals_df.columns)} columns")
        logger.info(f"Columns: {sorted(fundamentals_df.columns.tolist())}")

        # Save the COMPLETE fundamentals file with ALL columns
        fundamentals_path = self.output_dir / "fundamentals_all.parquet"
        fundamentals_df.to_parquet(fundamentals_path, index=False)
        outputs["fundamentals_all"] = fundamentals_path
        logger.info(f"Saved complete fundamentals to {fundamentals_path}")

        # Also save split files for backward compatibility, but include ALL relevant columns
        # Identify columns by checking what's in the data
        all_cols = set(fundamentals_df.columns)
        meta_cols = {"code", "date", "filing_date", "currency", "fiscalYear", "fiscalQuarter"}

        # Income statement - anything with revenue, income, expense, profit, etc.
        income_keywords = ["revenue", "income", "expense", "profit", "loss", "tax", "earning",
                          "ebit", "depreciation", "amortization", "research", "cost", "margin",
                          "interest", "dividend", "eps", "share"]
        income_cols = [c for c in all_cols if any(kw in c.lower() for kw in income_keywords)]
        income_cols = list(meta_cols & all_cols) + sorted(income_cols)

        if income_cols:
            income_df = fundamentals_df[[c for c in income_cols if c in all_cols]].copy()
            income_path = self.output_dir / EODHD_FILES["income"]
            income_df.to_parquet(income_path, index=False)
            outputs["income"] = income_path
            logger.info(f"Saved {len(income_df.columns)} income columns")

        # Balance sheet - assets, liabilities, equity, debt, cash, inventory, etc.
        balance_keywords = ["asset", "liabilit", "equity", "debt", "cash", "inventory",
                           "receivable", "payable", "capital", "stock", "share", "book",
                           "property", "plant", "equipment", "intangible", "goodwill",
                           "investment", "retain", "treasury"]
        balance_cols = [c for c in all_cols if any(kw in c.lower() for kw in balance_keywords)]
        balance_cols = list(meta_cols & all_cols) + sorted(balance_cols)

        if balance_cols:
            balance_df = fundamentals_df[[c for c in balance_cols if c in all_cols]].copy()
            balance_path = self.output_dir / EODHD_FILES["balance"]
            balance_df.to_parquet(balance_path, index=False)
            outputs["balance"] = balance_path
            logger.info(f"Saved {len(balance_df.columns)} balance columns")

        # Cash flow - operating, investing, financing activities
        cashflow_keywords = ["cashflow", "cash flow", "operating", "investing", "financing",
                            "capex", "capital expenditure", "dividend", "repurchase",
                            "issuance", "free cash"]
        cashflow_cols = [c for c in all_cols if any(kw in c.lower() for kw in cashflow_keywords)]
        cashflow_cols = list(meta_cols & all_cols) + sorted(cashflow_cols)

        if cashflow_cols:
            cashflow_df = fundamentals_df[[c for c in cashflow_cols if c in all_cols]].copy()
            cashflow_path = self.output_dir / EODHD_FILES["cashflow"]
            cashflow_df.to_parquet(cashflow_path, index=False)
            outputs["cashflow"] = cashflow_path
            logger.info(f"Saved {len(cashflow_df.columns)} cashflow columns")


def download_eodhd_data(
    api_key: Optional[str] = None,
    output_dir: Optional[str] = None,
    include_macro: bool = False,
    common_stocks_only: bool = True,
    max_symbols: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Convenience function to download EODHD data.

    Args:
        api_key: EODHD API key
        output_dir: Output directory
        include_macro: Download macro data
        common_stocks_only: Filter to common stocks
        max_symbols: Limit symbols for testing

    Returns:
        Dictionary of data type -> file path
    """
    output_path = Path(output_dir) if output_dir else EODHD_DIR
    downloader = EODHDDownloader(api_key=api_key, output_dir=output_path)

    return downloader.download_all(
        include_macro=include_macro,
        common_stocks_only=common_stocks_only,
        max_symbols=max_symbols,
    )
