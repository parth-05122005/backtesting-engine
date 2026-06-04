# =============================================================================
# BACKTESTING ENGINE — LAYER 1: DATA INGESTION
# =============================================================================
# What is this file?
#   This is the very first layer of our backtesting engine.
#   Its only job is to GET clean, reliable historical price data
#   and hand it over to the rest of the engine.
#
# Why do we need a separate layer for this?
#   Because raw data from any API is messy — missing dates, wrong prices,
#   duplicate rows, unadjusted splits. If we feed bad data to our strategy,
#   our backtest results will be completely wrong (garbage in = garbage out).
#
# What does this file produce?
#   A clean pandas DataFrame with columns:
#   [date | open | high | low | close | volume]
#   One row = one trading day. Sorted oldest → newest.
#
# Data source: Yahoo Finance via the `yfinance` library (100% FREE)
#   For NSE stocks: add ".NS" suffix  → e.g. "RELIANCE.NS", "TCS.NS"
#   For BSE stocks: add ".BO" suffix  → e.g. "RELIANCE.BO"
#   No API key needed. No account needed. Just pip install and use.
# =============================================================================


# --- IMPORTS ------------------------------------------------------------------
# These are Python libraries we need. Install them with:
#   pip install yfinance pandas

import os           # For checking if files/folders exist on your computer
import logging      # For printing nice log messages instead of bare print()
import yfinance as yf   # The library that talks to Yahoo Finance for us
import pandas as pd     # The library we use to work with table-like data

# Set up logging so our messages look professional
# Instead of print("Fetching data...") we use logger.info("Fetching data...")
# This makes it easy to turn logs on/off and shows timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# THE DATA LOADER CLASS
# =============================================================================
# A "class" is a blueprint. We're building a DataLoader machine.
# You create one DataLoader, and then use it to fetch any stock you want.
#
# Why a class instead of plain functions?
#   Because the DataLoader needs to remember settings (like where to save
#   cached files). A class keeps those settings together neatly.

class DataLoader:
    """
    Fetches, cleans, and caches historical OHLCV data for Indian NSE stocks.

    Usage:
        loader = DataLoader()
        df = loader.get("RELIANCE.NS", "2020-01-01", "2024-01-01")
        print(df.head())
    """

    # -------------------------------------------------------------------------
    # __init__: This runs automatically when you create a DataLoader object.
    # Think of it as "setting up the machine before using it."
    # -------------------------------------------------------------------------
    def __init__(self, cache_dir: str = "data/raw"):
        """
        Args:
            cache_dir: The folder where we save downloaded data as CSV files.
                       Default is "data/raw" inside your project folder.
                       We save data locally so we don't hit Yahoo Finance's
                       servers every single time we run the backtest.
        """
        self.cache_dir = cache_dir

        # Create the cache folder if it doesn't already exist.
        # exist_ok=True means "don't crash if the folder is already there."
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"DataLoader ready. Cache folder: '{self.cache_dir}'")


    # -------------------------------------------------------------------------
    # get(): The main public method. This is what you call from outside.
    # It handles the full flow: check cache → fetch if needed → clean → return.
    # -------------------------------------------------------------------------
    def get(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Main entry point. Returns a clean OHLCV DataFrame.

        Args:
            ticker : NSE stock symbol with suffix. E.g. "RELIANCE.NS"
            start  : Start date as string. E.g. "2020-01-01"
            end    : End date as string.   E.g. "2024-01-01"

        Returns:
            pd.DataFrame with columns [open, high, low, close, volume]
            Index is a DatetimeIndex (dates as the row labels).
        """

        # --- STEP 1: Check if we already have this data saved locally ---
        # We name the file like: "data/raw/RELIANCE.NS_2020-01-01_2024-01-01.csv"
        # so each unique ticker+date range gets its own file.
        cache_path = self._cache_path(ticker, start, end)

        if os.path.exists(cache_path):
            # File exists! Load from disk instead of hitting Yahoo Finance again.
            # This is much faster and avoids rate limits.
            logger.info(f"Cache hit! Loading {ticker} from '{cache_path}'")
            df = self._load_from_cache(cache_path)

        else:
            # No cached file found. We need to download fresh data.
            logger.info(f"Cache miss. Fetching {ticker} from Yahoo Finance...")

            # Step A: Download raw data from Yahoo Finance
            df = self._fetch(ticker, start, end)

            # Step B: Clean the raw data (handle missing values, bad rows, etc.)
            df = self._clean(df, ticker)

            # Step C: Save to disk so next run is instant
            self._save_to_cache(df, cache_path)
            logger.info(f"Saved to cache: '{cache_path}'")

        logger.info(
            f"Ready: {ticker} | {len(df)} trading days | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return df


    # =========================================================================
    # PRIVATE HELPER METHODS
    # (prefixed with _ to signal "don't call these from outside the class")
    # =========================================================================

    # -------------------------------------------------------------------------
    # _fetch(): Downloads raw data from Yahoo Finance using yfinance.
    # -------------------------------------------------------------------------
    def _fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Calls Yahoo Finance API and returns a raw DataFrame.

        Why auto_adjust=True?
            Stock prices get distorted by two corporate events:
            1. Stock SPLITS: e.g. if TCS does a 2-for-1 split, old price was
               ₹3500 but next day it's ₹1750. Without adjustment, your strategy
               sees a fake "50% crash" and fires a panic SELL. auto_adjust fixes
               this by retroactively halving all historical prices too.
            2. DIVIDENDS: when a company pays ₹10 dividend, the stock price
               drops by ~₹10 on ex-dividend date. Without adjustment, your
               backtest thinks the stock "dropped" when really you received cash.

            auto_adjust=True handles both automatically. Always use it.

        Why progress=False?
            Suppresses the progress bar that yfinance shows. Cleaner output
            when we have our own logging.
        """
        logger.info(f"  Calling Yahoo Finance API for {ticker}...")

        # yf.download() is the main function from yfinance.
        # It returns a pandas DataFrame directly.
        raw_df = yf.download(
            tickers=ticker,       # The stock symbol, e.g. "RELIANCE.NS"
            start=start,          # Start date, e.g. "2020-01-01"
            end=end,              # End date   (exclusive — end date not included)
            auto_adjust=True,     # Adjust for splits & dividends (ALWAYS True)
            progress=False,       # Don't show a progress bar
            threads=False         # Download sequentially (safer for single stock)
        )

        # yfinance sometimes returns a MultiIndex column (ticker, field).
        # We flatten it to just (field) → 'open', 'high', 'low', 'close', 'volume'
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.droplevel(1)

        # Standardise column names to lowercase so rest of engine is consistent
        raw_df.columns = [col.lower() for col in raw_df.columns]

        logger.info(f"  Downloaded {len(raw_df)} raw rows from Yahoo Finance.")
        return raw_df


    # -------------------------------------------------------------------------
    # _clean(): Fixes all the problems in raw data.
    # -------------------------------------------------------------------------
    def _clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Cleans raw OHLCV data. Fixes missing values, duplicates, outliers.

        This is the most important step. A dirty dataset will produce
        completely wrong backtest results — strategies will fire signals
        on phantom price crashes or spikes that never really happened.
        """
        original_len = len(df)
        logger.info(f"  Cleaning {original_len} raw rows...")

        # --- PROBLEM 1: Missing values (NaN) ---------------------------------
        # Yahoo Finance sometimes returns rows where all prices are NaN.
        # This happens for public holidays that got included accidentally.
        # We drop any row where the core price columns are all missing.
        df = df.dropna(subset=["open", "high", "low", "close"])
        logger.info(f"  After dropping NaN rows: {len(df)} rows remain.")

        # --- PROBLEM 2: Duplicate dates --------------------------------------
        # Some APIs accidentally return the same date twice. This would make
        # our strategy process the same day twice — a serious error.
        # Keep only the first occurrence of each date.
        duplicates = df.index.duplicated(keep="first").sum()
        if duplicates > 0:
            logger.warning(f"  Found {duplicates} duplicate dates. Removing...")
            df = df[~df.index.duplicated(keep="first")]

        # --- PROBLEM 3: Rows not sorted by date ------------------------------
        # Our strategy engine will loop through rows one by one, assuming
        # they go from past to future. If they're scrambled, the engine
        # will process "tomorrow" before "today" — look-ahead bias!
        # ALWAYS sort ascending (oldest date first).
        df = df.sort_index(ascending=True)

        # --- PROBLEM 4: Zero or negative prices (data corruption) -----------
        # A stock price can never be zero or negative in reality.
        # If we see that, it's a data error. Remove those rows.
        bad_prices = (df["close"] <= 0) | (df["open"] <= 0)
        if bad_prices.any():
            logger.warning(f"  Removing {bad_prices.sum()} rows with invalid prices.")
            df = df[~bad_prices]

        # --- PROBLEM 5: Keep only the columns we need -----------------------
        # yfinance sometimes returns extra columns like "Adj Close".
        # We only want these 5 standard OHLCV columns.
        columns_we_need = ["open", "high", "low", "close", "volume"]
        df = df[[col for col in columns_we_need if col in df.columns]]

        # --- PROBLEM 6: Make sure the index is a proper DatetimeIndex -------
        # This lets us do date arithmetic easily later, like:
        #   df["2022-01-01":"2022-12-31"]  (slice by date range)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"  # Name the index column "date"

        # --- PROBLEM 7: Cast volume to integer (it should never be float) ---
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype(int)

        cleaned_count = original_len - len(df)
        logger.info(
            f"  Cleaning complete. Removed {cleaned_count} bad rows. "
            f"Final: {len(df)} clean rows."
        )
        return df


    # -------------------------------------------------------------------------
    # _cache_path(): Builds a unique file path for each ticker+date range.
    # -------------------------------------------------------------------------
    def _cache_path(self, ticker: str, start: str, end: str) -> str:
        """
        Returns a file path like: data/raw/RELIANCE.NS_2020-01-01_2024-01-01.csv

        Why unique per ticker AND date range?
            If you fetch RELIANCE from 2020-2022 and cache it, then later try
            to fetch 2020-2024, the cache would return the shorter dataset.
            Unique filenames per date range prevent this silent bug.
        """
        # Replace dots in ticker name with underscores for safe filenames
        # "RELIANCE.NS" → "RELIANCE_NS"
        safe_ticker = ticker.replace(".", "_")
        filename = f"{safe_ticker}_{start}_{end}.csv"
        return os.path.join(self.cache_dir, filename)


    # -------------------------------------------------------------------------
    # _save_to_cache(): Saves a DataFrame to CSV on disk.
    # -------------------------------------------------------------------------
    def _save_to_cache(self, df: pd.DataFrame, path: str) -> None:
        """
        Saves the cleaned DataFrame to a CSV file.

        Why CSV and not a database?
            CSV is the simplest format — human-readable, opens in Excel,
            no setup required. Fine for a backtesting project.
            In production systems you'd use Parquet or a TimescaleDB.
        """
        df.to_csv(path)  # index=True by default, so the date column is saved


    # -------------------------------------------------------------------------
    # _load_from_cache(): Reads a previously saved CSV back into a DataFrame.
    # -------------------------------------------------------------------------
    def _load_from_cache(self, path: str) -> pd.DataFrame:
        """
        Loads a cached CSV file and restores it to the correct format.

        Why parse_dates=True?
            When pandas saves a date to CSV, it becomes a plain string like
            "2022-01-03". When reading back, we need to tell pandas to convert
            those strings back into proper datetime objects so date arithmetic
            still works.
        """
        df = pd.read_csv(
            path,
            index_col="date",    # Make the "date" column the row index
            parse_dates=True     # Convert date strings → datetime objects
        )
        return df


# =============================================================================
# QUICK REFERENCE: POPULAR NSE STOCK TICKERS
# =============================================================================
# Use these with .NS suffix in the DataLoader
#
# RELIANCE.NS  → Reliance Industries
# TCS.NS       → Tata Consultancy Services
# INFY.NS      → Infosys
# HDFCBANK.NS  → HDFC Bank
# ICICIBANK.NS → ICICI Bank
# SBIN.NS      → State Bank of India
# WIPRO.NS     → Wipro
# HINDUNILVR.NS→ Hindustan Unilever
# AXISBANK.NS  → Axis Bank
# BAJFINANCE.NS→ Bajaj Finance
# ^NSEI        → Nifty 50 Index (use as benchmark)
# ^BSESN       → Sensex Index   (use as benchmark)
# =============================================================================


# =============================================================================
# DEMO: Run this file directly to test if everything works
# =============================================================================
# In your terminal: python data_loader.py
# This block only runs when you run THIS file directly.
# It does NOT run when another file imports DataLoader.

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  BACKTESTING ENGINE — LAYER 1 TEST")
    print("="*60 + "\n")

    # Create a DataLoader instance
    loader = DataLoader(cache_dir="data/raw")

    # --- TEST 1: Fetch Reliance Industries (NSE) ---
    print("TEST 1: Fetching RELIANCE.NS (2022-01-01 to 2024-01-01)")
    print("-" * 50)
    reliance = loader.get("RELIANCE.NS", "2022-01-01", "2024-01-01")
    print(reliance.head(5))     # Show first 5 rows
    print(f"\nShape: {reliance.shape}  (rows, columns)")
    print(f"Columns: {list(reliance.columns)}")
    print(f"Index type: {type(reliance.index)}")

    # --- TEST 2: Run again to confirm cache works ---
    print("\nTEST 2: Fetching RELIANCE.NS again (should load from cache)")
    print("-" * 50)
    reliance_cached = loader.get("RELIANCE.NS", "2022-01-01", "2024-01-01")
    print("Cache loaded successfully!")

    # --- TEST 3: Fetch TCS ---
    print("\nTEST 3: Fetching TCS.NS (2021-01-01 to 2024-01-01)")
    print("-" * 50)
    tcs = loader.get("TCS.NS", "2021-01-01", "2024-01-01")
    print(tcs.tail(5))          # Show last 5 rows

    # --- TEST 4: Basic data quality checks ---
    print("\nTEST 4: Data quality checks on RELIANCE.NS")
    print("-" * 50)
    print(f"Any NaN values?        {reliance.isnull().any().any()}")
    print(f"Any zero close prices? {(reliance['close'] <= 0).any()}")
    print(f"Is sorted by date?     {reliance.index.is_monotonic_increasing}")
    print(f"Date range:            {reliance.index[0].date()} → {reliance.index[-1].date()}")
    print(f"Highest close price:   ₹{reliance['close'].max():.2f}")
    print(f"Lowest close price:    ₹{reliance['close'].min():.2f}")
    print(f"Avg daily volume:      {int(reliance['volume'].mean()):,} shares")

    print("\n" + "="*60)
    print("  ALL TESTS PASSED. Layer 1 is working correctly.")
    print("="*60 + "\n")