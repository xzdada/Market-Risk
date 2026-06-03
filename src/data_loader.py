"""
Downloads and saves raw market data (equity prices) and macroeconomic indicators (FRED) to data/raw/. 
All further cleaning and feature engineering happens in Notebook 01.

Usage
-----
    python src/data_loader.py --fred-key FRED_KEY

    OR
        
    from src.data_loader import DataLoader
    loader = DataLoader(fred_api_key="FRED_KEY")
    loader.run()
"""
from dotenv import load_dotenv
import os
load_dotenv()
fred_key = os.getenv("FRED_API_KEY")

import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

START_DATE = "2005-01-01"
END_DATE   = "2026-01-01"

# 12 equities across 6 sectors
EQUITY_TICKERS = {
    "Technology": ["AAPL", "MSFT"],
    "Financials": ["JPM",  "GS"  ],
    "Energy": ["XOM",  "CVX" ],
    "Healthcare": ["JNJ",  "UNH" ],
    "Consumer": ["PG",   "KO"  ],
    "Industrials": ["HON",  "CAT" ],
}

# FRED series
FRED_SERIES = {
    "USREC": "nber_recession", # NBER recession indicator (0/1, monthly)
    "VIXCLS": "vix", # CBOE VIX (daily)
    "DFF": "fed_funds_rate", # Effective Fed Funds Rate (daily)
    "CPIAUCSL": "cpi",  # CPI all items (monthly)
    "T10Y2Y": "yield_curve_spread",  # 10Y-2Y Treasury spread (daily)
    "T10YIE": "breakeven_inflation",  # 10Y breakeven inflation (daily)
}

# output paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PRICES = ROOT_DIR / "data" / "raw" / "prices"
RAW_MACRO = ROOT_DIR / "data" / "raw" / "macro"

# Helper utilities
def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in [RAW_PRICES, RAW_MACRO]:
        d.mkdir(parents=True, exist_ok=True)

def _data_quality_report(df: pd.DataFrame, name: str) -> dict:
    """
    Print and return a quality report for a DataFrame.
    Returns a dict with keys: n_rows, n_cols, missing_pct (per column), date_range, n_duplicates.
    """
    missing = (df.isna().sum() / len(df) * 100).round(2)
    report = {
        "name": name,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "date_range": f"{df.index.min().date()} -> {df.index.max().date()}",
        "n_duplicates": int(df.index.duplicated().sum()),
        "missing_pct": missing.to_dict(),
    }

    log.info("─" * 60)
    log.info(f"Quality report: {name}")
    log.info(f"Rows: {report['n_rows']:,}")
    log.info(f"Columns: {report['n_cols']}")
    log.info(f"Date range: {report['date_range']}")
    log.info(f"Duplicates: {report['n_duplicates']}")

    # Warn if any column exceeds 5% missing
    bad_cols = {c: v for c, v in missing.items() if v > 5}
    if bad_cols:
        log.warning(f"High missing (>5%): {bad_cols}")
    else:
        max_missing = missing.max()
        log.info(f"Max missing %  : {max_missing:.2f}% ")

    return report

# Equity data

def download_equity_prices(tickers: dict[str, list[str]], 
                           start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download adjusted close prices for all tickers via yfinance.

    Parameters
    ----------
    tickers : dict mapping sector -> list of ticker strings
    start, end : date strings "YYYY-MM-DD"

    Returns
    -------
    prices_wide : DataFrame, shape (T, N), columns = ticker symbols
    metadata : DataFrame with ticker, sector, first_date, last_date, missing_pct columns
    """
    all_tickers = [t for tickers_list in tickers.values() for t in tickers_list]
    sector_map  = {t: s for s, tlist in tickers.items() for t in tlist}

    log.info(f"Downloading prices for {len(all_tickers)} tickers " f"({start} -> {end}) ...")

    raw = yf.download(
        tickers = all_tickers,
        start = start,
        end = end,
        auto_adjust= True,
        progress = False,
    )

    # Extract adjusted close; handle single vs multi ticker response
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single ticker edge case
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    # Sort columns alphabetically for reproducibility
    prices = prices.sort_index(axis=1)

    meta_rows = []
    for ticker in all_tickers:
        if ticker not in prices.columns:
            log.warning(f"  {ticker} not found in download — skipping")
            continue
        col = prices[ticker].dropna()
        missing_pct = prices[ticker].isna().mean() * 100
        meta_rows.append({
            "ticker" : ticker,
            "sector" : sector_map[ticker],
            "first_date" : col.index.min().date() if len(col) else None,
            "last_date" : col.index.max().date() if len(col) else None,
            "missing_pct": round(missing_pct, 2),
            "n_obs" : len(col),
        })

    metadata = pd.DataFrame(meta_rows).set_index("ticker")

    log.info("Per-ticker summary:")
    for tkr, row in metadata.iterrows():
        flag = "  ⚠" if row["missing_pct"] > 5 else "  ✓"
        log.info(f"{flag} {tkr:<6}  {row['sector']:<12}  "
                 f"{row['first_date']} -> {row['last_date']}  "
                 f"missing={row['missing_pct']:.1f}%")

    _data_quality_report(prices, "equity_prices")
    return prices, metadata

# Macro / FRED data

def download_fred_series(
    series: dict[str, str],
    fred_api_key: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download multiple FRED series and return as a single wide DataFrame.
    """
    fred   = Fred(api_key=fred_api_key)
    frames = {}

    for series_id, col_name in series.items():
        log.info(f"  Fetching FRED series: {series_id} ({col_name}) ...")
        try:
            s = fred.get_series(
                series_id,
                observation_start=start,
                observation_end  =end,
            )
            s.name = col_name
            s.index = pd.to_datetime(s.index)
            s.index.name = "date"
            frames[col_name] = s
            log.info(f"  {len(s):,} observations, {s.index.min().date()} → {s.index.max().date()}")
        except Exception as exc:
            log.error(f"  Failed to fetch {series_id}: {exc}")

    if not frames:
        raise RuntimeError("No FRED series downloaded successfully.")

    macro = pd.DataFrame(frames)
    macro.sort_index(inplace=True)

    _data_quality_report(macro, "macro_fred")
    return macro

# Save helpers

def _save_csv(df: pd.DataFrame, path: Path, name: str) -> None:
    """Save DataFrame to CSV and log the outcome."""
    df.to_csv(path)
    size_kb = path.stat().st_size / 1024
    log.info(f"Saved {name} -> {path.relative_to(ROOT_DIR)}  " f"({size_kb:.1f} KB, {len(df):,} rows)")


# ---------------------------------------------------------------------------
# Main entry point

class DataLoader:
    """
    Orchestrates all data downloads for the portfolio risk system.

    Parameters
    ----------
    fred_api_key : str
    start, end   : str, Date range in "YYYY-MM-DD" format.
    tickers : dict, optional, Override the default EQUITY_TICKERS dict.
    fred_series  : dict, optional, Override the default FRED_SERIES dict.
    """

    def __init__(
        self,
        fred_api_key: str,
        start: str = START_DATE,
        end: str = END_DATE,
        tickers: dict | None = None,
        fred_series: dict | None = None,
    ):
        self.fred_api_key = fred_api_key
        self.start = start
        self.end = end
        self.tickers = tickers or EQUITY_TICKERS
        self.fred_series = fred_series or FRED_SERIES

    def run(self) -> dict[str, pd.DataFrame]:
        """
        Run all downloads and save raw files.
        """
        _ensure_dirs()
        log.info("=" * 60)
        log.info("Portfolio Risk System — Data Loader")
        log.info(f"Date range : {self.start} → {self.end}")
        log.info("=" * 60)

        # Equity prices
        log.info("\n[1/2] Equity prices (yfinance)")
        prices, metadata = download_equity_prices(self.tickers, self.start, self.end)
        _save_csv(prices, RAW_PRICES / "equity_prices.csv",    "equity_prices")
        _save_csv(metadata, RAW_PRICES / "equity_metadata.csv",  "equity_metadata")

        # Macro indicators
        log.info("\n[2/2] Macro indicators (FRED)")
        macro = download_fred_series(self.fred_series, self.fred_api_key, self.start, self.end)
        _save_csv(macro, RAW_MACRO / "macro_fred.csv", "macro_fred")

        # Final summary
        log.info("\n" + "=" * 60)
        log.info("Download complete.")
        log.info(f"  Equity tickers : {prices.shape[1]}")
        log.info(f"  Equity obs : {prices.shape[0]:,} trading days")
        log.info(f"  Macro series : {macro.shape[1]}")
        log.info(f"  Macro obs : {macro.shape[0]:,} calendar days")
        log.info(f"  Raw files : {RAW_PRICES}  |  {RAW_MACRO}")
        log.info("=" * 60)

        return {"prices": prices, "metadata": metadata, "macro": macro}

# CLI entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download raw market and macro data for the portfolio risk system."
    )
    parser.add_argument(
        "--fred-key",
        type=str,
        default=None,
        help="Your FRED API key. If omitted, uses FRED_API_KEY from .env",
    )
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end",   type=str, default=END_DATE)
    args = parser.parse_args()

    fred_api_key = args.fred_key or os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise ValueError("FRED API key not found. Pass --fred-key or set FRED_API_KEY in .env")

    loader = DataLoader(
        fred_api_key=fred_api_key,
        start=args.start,
        end=args.end,
    )
    loader.run()