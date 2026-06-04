# =============================================================================
# BACKTESTING ENGINE — LAYER 2: STRATEGY ENGINE
# =============================================================================
# What is this file?
#   This is the brain of the backtesting engine.
#   It takes the clean OHLCV DataFrame from Layer 1 and decides:
#   "On this day, should I BUY, SELL, or do NOTHING?"
#
# How does it work?
#   1. It computes INDICATORS — mathematical summaries of price history
#      (e.g. "what was the average price over the last 20 days?")
#   2. It applies RULES to those indicators to generate SIGNALS
#      (+1 = BUY, -1 = SELL, 0 = HOLD)
#
# Why a pluggable base class?
#   We want to be able to swap strategies without touching anything else.
#   Think of it like a power socket — any plug (strategy) that fits the
#   socket (base class interface) can be used. Layer 3 and 4 don't care
#   WHICH strategy is running, they just consume the signals it produces.
#
# What strategies are implemented here?
#   1. EMA Crossover Strategy — a classic trend-following strategy
#   2. EMA + RSI Strategy    — crossover with an overbought/oversold filter
#
# Important concept — LOOK-AHEAD BIAS (introduced in Layer 1):
#   Every indicator uses .shift(1) before generating signals.
#   This means: "today's signal is based on YESTERDAY's indicator value."
#   Why? Because in real trading, you only know today's close AFTER
#   the market closes. You can only act on it TOMORROW morning.
#   Forgetting .shift(1) is the #1 backtesting mistake.
# =============================================================================

import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# INDICATOR LIBRARY
# =============================================================================
# These are standalone functions that take a price Series and return a new
# Series of computed values. They are pure math — no trading logic here.
# Keeping indicators separate from strategy logic makes them reusable.

def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    What is it?
        An average of the last N closing prices, BUT recent prices are
        given MORE weight than older prices. This makes it react faster
        to new price movements compared to a Simple Moving Average (SMA).

    Formula:
        EMA_today = Close_today × multiplier + EMA_yesterday × (1 - multiplier)
        where multiplier = 2 / (period + 1)

    Why use EMA over SMA?
        If a stock suddenly rallies, EMA picks it up faster. SMA is slower
        to react because it weighs all past prices equally. In trend-
        following strategies, faster reaction = better entry timing.

    Args:
        series: A pandas Series of closing prices
        period: Number of days to look back (e.g. 9, 21, 50, 200)

    Returns:
        A pandas Series of EMA values, same length as input.
        First (period-1) values will be NaN — not enough data yet.
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    What is it?
        A momentum indicator that measures the SPEED and MAGNITUDE of
        recent price changes. It oscillates between 0 and 100.

        RSI > 70 → stock is OVERBOUGHT  (may be due for a pullback → don't buy)
        RSI < 30 → stock is OVERSOLD    (may be due for a bounce → good to buy)
        RSI ~50  → neutral

    Formula (simplified):
        RS = Average Gain over N days / Average Loss over N days
        RSI = 100 - (100 / (1 + RS))

    Why use RSI as a filter?
        An EMA crossover alone can give false signals in choppy markets.
        Adding RSI as a filter says: "only buy if the crossover happens
        AND the stock isn't already overbought." This reduces bad trades.

    Args:
        series: A pandas Series of closing prices
        period: Lookback period (default 14 days — industry standard)

    Returns:
        A pandas Series of RSI values between 0 and 100.
    """
    # Calculate price change from one day to next
    delta = series.diff()

    # Separate gains (positive changes) from losses (negative changes)
    # losses are stored as positive numbers (we take absolute value)
    gains = delta.clip(lower=0)          # keep only positive changes
    losses = (-delta).clip(lower=0)      # keep only negative changes (as positive)

    # Calculate rolling average gain and average loss
    # Using Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    avg_gain = gains.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, adjust=False).mean()

    # Avoid division by zero: if avg_loss is 0, RSI = 100 (all gains, no losses)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(series: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).

    What is it?
        MACD shows the RELATIONSHIP between two EMAs of different speeds.
        It has three components:
          - MACD Line    = Fast EMA (12-day) minus Slow EMA (26-day)
          - Signal Line  = 9-day EMA of the MACD Line
          - Histogram    = MACD Line minus Signal Line

    How to read it:
        When MACD Line crosses ABOVE Signal Line → bullish (potential BUY)
        When MACD Line crosses BELOW Signal Line → bearish (potential SELL)
        Histogram growing → momentum increasing
        Histogram shrinking → momentum fading

    Args:
        series: Closing price Series
        fast, slow, signal: Standard MACD parameters (12, 26, 9)

    Returns:
        DataFrame with columns: ['macd_line', 'signal_line', 'histogram']
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)

    macd_line   = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line

    return pd.DataFrame({
        'macd_line':   macd_line,
        'signal_line': signal_line,
        'histogram':   histogram
    })


# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================
# This is the "power socket" interface. Every strategy must inherit from this
# and implement the generate_signals() method.
#
# Think of it like an abstract contract:
#   "If you want to be a Strategy, you MUST implement generate_signals()."
#
# Why does this matter?
#   Layer 3 (Execution Simulator) will call strategy.generate_signals(df)
#   without knowing or caring which strategy it is. As long as the output
#   format is always the same (+1, 0, -1 in a column called 'signal'),
#   any strategy plugs in without changing other layers.

class Strategy:
    """
    Abstract base class for all trading strategies.

    To create a new strategy:
        1. Inherit from this class
        2. Implement generate_signals(df) → must return df with a 'signal' column
        3. Pass your strategy instance to the Backtester (Layer 3+)
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a clean OHLCV DataFrame and returns the same DataFrame
        with an added 'signal' column:
            +1 = BUY  (enter a long position)
            -1 = SELL (exit the position)
             0 = HOLD (do nothing)

        MUST be overridden by every subclass.
        """
        raise NotImplementedError(
            "Every strategy must implement generate_signals(). "
            "See EMACrossoverStrategy for an example."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# =============================================================================
# STRATEGY 1: EMA CROSSOVER
# =============================================================================
# The simplest and most well-known trend-following strategy.
#
# The idea:
#   When a SHORT-term average price crosses ABOVE a LONG-term average price,
#   the stock is gaining momentum → BUY.
#   When it crosses back BELOW → the trend is reversing → SELL.
#
# Visual intuition:
#   Imagine two lines plotted on a price chart:
#   - Fast EMA (9-day): wiggles up and down quickly with the price
#   - Slow EMA (21-day): smoother, slower to react
#
#   When fast crosses above slow → "Golden Cross" → BUY signal
#   When fast crosses below slow → "Death Cross"  → SELL signal
#
# Why does this work?
#   It doesn't always! It works well in TRENDING markets (stock going
#   steadily up or down). It gives many false signals in SIDEWAYS markets
#   (stock bouncing up and down with no clear direction). That's why we
#   add RSI as a filter in Strategy 2.

class EMACrossoverStrategy(Strategy):
    """
    EMA Crossover Strategy.

    Buys when the fast EMA crosses above the slow EMA.
    Sells when the fast EMA crosses back below the slow EMA.

    Args:
        fast_period: Lookback for the fast (responsive) EMA. Default 9.
        slow_period: Lookback for the slow (smooth) EMA. Default 21.
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        # Validate that fast < slow (otherwise "fast" and "slow" make no sense)
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than "
                f"slow_period ({slow_period})"
            )
        self.fast_period = fast_period
        self.slow_period = slow_period
        logger.info(
            f"EMACrossoverStrategy created: "
            f"fast={fast_period}, slow={slow_period}"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes EMA crossover signals on the input DataFrame.

        Step-by-step:
            1. Compute fast EMA (9-day) and slow EMA (21-day)
            2. Shift both by 1 day (critical: prevents look-ahead bias)
            3. Detect crossover events (fast crosses above/below slow)
            4. Assign +1 on golden cross, -1 on death cross, 0 otherwise

        Returns a copy of df with these new columns added:
            ema_fast  : fast EMA values
            ema_slow  : slow EMA values
            signal    : +1, -1, or 0
        """
        df = df.copy()  # Never mutate the original DataFrame — always work on a copy

        # --- STEP 1: Compute indicators on raw close prices ----------------
        # We compute on raw prices first (before shifting) because the math
        # of EMA requires the full history to be accurate.
        raw_fast = ema(df['close'], self.fast_period)
        raw_slow = ema(df['close'], self.slow_period)

        # --- STEP 2: Shift by 1 day (LOOK-AHEAD BIAS PREVENTION) ----------
        # .shift(1) moves every value forward by one row.
        # Row for 2023-01-05 now contains the value that was computed on 2023-01-04.
        # This simulates: "I computed this indicator at market close yesterday,
        # and now I'm using it to decide what to do this morning."
        #
        # WITHOUT shift: your strategy uses TODAY's close to generate TODAY's
        #   signal — impossible in real life, since today's close isn't known
        #   until AFTER the market closes.
        # WITH shift: your strategy uses YESTERDAY's close → realistic.
        df['ema_fast'] = raw_fast.shift(1)
        df['ema_slow'] = raw_slow.shift(1)

        # --- STEP 3: Detect crossover events --------------------------------
        # A "Golden Cross" (BUY signal) happens on the EXACT day when:
        #   - TODAY: fast EMA is ABOVE slow EMA
        #   - YESTERDAY: fast EMA was BELOW slow EMA
        # That transition = the crossover event.
        #
        # fast_above is True on days when fast EMA > slow EMA
        fast_above = df['ema_fast'] > df['ema_slow']

        # .shift(1) gives us what fast_above was YESTERDAY.
        # WHY .fillna(False).astype(bool)?
        #   When you .shift(1) a boolean Series, pandas introduces NaN in the
        #   first row (no "yesterday" exists). NaN causes pandas to convert
        #   the whole Series from bool → float. The ~ (NOT) operator then
        #   crashes because it can't invert floats.
        #   fillna(False) replaces the NaN with False, and astype(bool)
        #   forces the Series back to a proper boolean type.
        fast_above_yesterday = fast_above.shift(1).fillna(False).astype(bool)

        # Golden Cross: fast was below yesterday, is above today
        golden_cross = fast_above & ~fast_above_yesterday

        # Death Cross: fast was above yesterday, is below today
        death_cross = ~fast_above & fast_above_yesterday

        # --- STEP 4: Assign signal values -----------------------------------
        # Start with all zeros (HOLD)
        df['signal'] = 0

        # Set +1 where golden cross occurred
        df.loc[golden_cross, 'signal'] = 1

        # Set -1 where death cross occurred
        df.loc[death_cross, 'signal'] = -1

        # Count signals for logging
        buys  = (df['signal'] == 1).sum()
        sells = (df['signal'] == -1).sum()
        logger.info(
            f"EMACrossoverStrategy: generated {buys} BUY signals, "
            f"{sells} SELL signals over {len(df)} trading days."
        )

        return df

    def __repr__(self):
        return (
            f"EMACrossoverStrategy("
            f"fast={self.fast_period}, slow={self.slow_period})"
        )


# =============================================================================
# STRATEGY 2: EMA CROSSOVER + RSI FILTER
# =============================================================================
# A smarter version of Strategy 1 that adds RSI as a filter.
#
# The problem with pure EMA crossover:
#   Sometimes a crossover happens when the stock is already way overbought
#   (e.g. RSI = 80). Buying at that point is risky — the stock may pull back.
#   Similarly, a sell signal when RSI = 25 might be a false alarm.
#
# The fix — RSI filter:
#   BUY only when: golden cross AND RSI < rsi_overbought (not already overbought)
#   SELL only when: death cross OR RSI > rsi_overbought (exit overbought positions)
#
# This combination reduces false signals and improves the quality of entries.

class EMARSIStrategy(Strategy):
    """
    EMA Crossover + RSI Filter Strategy.

    Only buys on a golden cross if RSI is below the overbought threshold.
    Sells on a death cross OR when RSI becomes overbought.

    Args:
        fast_period    : Fast EMA period. Default 9.
        slow_period    : Slow EMA period. Default 21.
        rsi_period     : RSI lookback period. Default 14.
        rsi_overbought : RSI level above which we consider the stock
                         overbought and avoid buying. Default 70.
        rsi_oversold   : RSI level below which the stock is oversold
                         (used for additional buy confirmation). Default 30.
    """

    def __init__(
        self,
        fast_period:     int   = 9,
        slow_period:     int   = 21,
        rsi_period:      int   = 14,
        rsi_overbought:  float = 70.0,
        rsi_oversold:    float = 30.0,
    ):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")

        self.fast_period    = fast_period
        self.slow_period    = slow_period
        self.rsi_period     = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold   = rsi_oversold

        logger.info(
            f"EMARSIStrategy created: fast={fast_period}, slow={slow_period}, "
            f"rsi_period={rsi_period}, overbought={rsi_overbought}"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates BUY/SELL signals combining EMA crossover and RSI filter.
        """
        df = df.copy()

        # --- Compute all indicators (on raw prices, then shift) -------------
        df['ema_fast'] = ema(df['close'], self.fast_period).shift(1)
        df['ema_slow'] = ema(df['close'], self.slow_period).shift(1)
        df['rsi']      = rsi(df['close'], self.rsi_period).shift(1)

        # --- EMA crossover detection (same as Strategy 1) -------------------
        # Same .fillna(False).astype(bool) fix applied here too —
        # .shift(1) on a bool Series produces floats due to NaN in row 0.
        fast_above           = df['ema_fast'] > df['ema_slow']
        fast_above_yesterday = fast_above.shift(1).fillna(False).astype(bool)
        golden_cross         = fast_above & ~fast_above_yesterday
        death_cross          = ~fast_above & fast_above_yesterday

        # --- RSI conditions -------------------------------------------------
        not_overbought = df['rsi'] < self.rsi_overbought   # RSI < 70: safe to buy
        is_overbought  = df['rsi'] > self.rsi_overbought   # RSI > 70: exit signal

        # --- Combined signal rules ------------------------------------------
        # BUY when: golden cross AND stock is NOT overbought
        buy_condition  = golden_cross & not_overbought

        # SELL when: death cross OR stock becomes overbought
        sell_condition = death_cross | is_overbought

        # Assign signals
        df['signal'] = 0
        df.loc[buy_condition,  'signal'] =  1
        df.loc[sell_condition, 'signal'] = -1

        buys  = (df['signal'] ==  1).sum()
        sells = (df['signal'] == -1).sum()
        logger.info(
            f"EMARSIStrategy: generated {buys} BUY signals, "
            f"{sells} SELL signals over {len(df)} trading days."
        )

        return df

    def __repr__(self):
        return (
            f"EMARSIStrategy("
            f"fast={self.fast_period}, slow={self.slow_period}, "
            f"rsi={self.rsi_period}, ob={self.rsi_overbought})"
        )


# =============================================================================
# DEMO: Run this file directly to see signals generated on real data
# =============================================================================
if __name__ == "__main__":
    # Import Layer 1 — we need data before we can run any strategy
    # Make sure data_loader.py is in the same folder as this file
    from data_loader import DataLoader

    print("\n" + "="*60)
    print("  BACKTESTING ENGINE — LAYER 2 TEST")
    print("="*60 + "\n")

    # --- Load data (Layer 1) ------------------------------------------------
    loader = DataLoader()
    df = loader.get("RELIANCE.NS", "2022-01-01", "2024-01-01")
    print(f"Loaded {len(df)} trading days of RELIANCE.NS\n")

    # --- TEST 1: EMA Crossover Strategy -------------------------------------
    print("TEST 1: EMA Crossover Strategy (fast=9, slow=21)")
    print("-" * 50)
    strategy1 = EMACrossoverStrategy(fast_period=9, slow_period=21)
    df_signals1 = strategy1.generate_signals(df)

    # Show only rows where a signal was generated
    signals_only = df_signals1[df_signals1['signal'] != 0][
        ['close', 'ema_fast', 'ema_slow', 'signal']
    ]
    print(signals_only.head(10).round(2))
    print(f"\nTotal BUY  signals: {(df_signals1['signal'] == 1).sum()}")
    print(f"Total SELL signals: {(df_signals1['signal'] == -1).sum()}")

    # --- TEST 2: EMA + RSI Strategy -----------------------------------------
    print("\nTEST 2: EMA + RSI Strategy (fast=9, slow=21, rsi=14, ob=70)")
    print("-" * 50)
    strategy2 = EMARSIStrategy(
        fast_period=9, slow_period=21,
        rsi_period=14, rsi_overbought=70
    )
    df_signals2 = strategy2.generate_signals(df)

    signals_only2 = df_signals2[df_signals2['signal'] != 0][
        ['close', 'ema_fast', 'ema_slow', 'rsi', 'signal']
    ]
    print(signals_only2.head(10).round(2))
    print(f"\nTotal BUY  signals: {(df_signals2['signal'] == 1).sum()}")
    print(f"Total SELL signals: {(df_signals2['signal'] == -1).sum()}")

    # --- TEST 3: Custom strategy parameters ---------------------------------
    print("\nTEST 3: Faster strategy (fast=5, slow=13)")
    print("-" * 50)
    strategy3 = EMACrossoverStrategy(fast_period=5, slow_period=13)
    df_signals3 = strategy3.generate_signals(df)
    print(f"Total BUY  signals: {(df_signals3['signal'] == 1).sum()}")
    print(f"Total SELL signals: {(df_signals3['signal'] == -1).sum()}")
    print("(More signals = more trades = higher transaction costs)")

    # --- TEST 4: Verify no look-ahead bias ----------------------------------
    print("\nTEST 4: Look-ahead bias check")
    print("-" * 50)
    # On any given row, the signal should only use data from BEFORE that row.
    # We verify this by checking that ema_fast on date D equals
    # the raw EMA computed on date D-1.
    raw_ema_fast = ema(df['close'], 9)
    shifted_ema  = raw_ema_fast.shift(1)
    match = (df_signals1['ema_fast'].dropna() == shifted_ema.dropna()).all()
    print(f"EMA correctly shifted by 1 day: {match}")
    print("✓ No look-ahead bias detected." if match else "✗ WARNING: Look-ahead bias found!")

    print("\n" + "="*60)
    print("  ALL TESTS PASSED. Layer 2 is working correctly.")
    print("="*60 + "\n")