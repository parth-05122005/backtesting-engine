# =============================================================================
# BACKTESTING ENGINE — LAYER 3: EXECUTION SIMULATOR
#                      LAYER 4: PORTFOLIO TRACKER
# =============================================================================
# Why are these two layers in one file?
#   They work as one unit. The Execution Simulator decides WHAT to trade
#   and at WHAT price. The Portfolio Tracker records the EFFECT of that trade
#   on your cash, positions, and total wealth. They talk to each other on
#   every single bar (every trading day).
#
# Layer 3 — Execution Simulator answers:
#   "My strategy said BUY on 2023-05-12. What price did I actually pay?
#    How much brokerage did I pay? How many shares can I actually afford?"
#
# Layer 4 — Portfolio Tracker answers:
#   "After that trade, how much cash do I have left? How many shares do I hold?
#    What is my total portfolio worth today?"
#
# Together they produce an EQUITY CURVE — a list of your portfolio value
# on every single trading day. This is what Layer 5 will analyse.
#
# Key concepts introduced here:
#   - Slippage   : In real markets, large orders move the price against you
#   - Brokerage  : The fee you pay each time you buy or sell
#   - Position   : The number of shares you currently hold
#   - Equity     : Cash + (shares × current price) = total portfolio value
#   - Equity curve: Equity value recorded every day → the main backtest output
# =============================================================================

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES — small containers to hold structured data cleanly
# =============================================================================

@dataclass
class Trade:
    """
    Represents a single completed trade (buy or sell).

    We record every trade so we can analyse them later:
    - How many trades were profitable?
    - What was the average profit per trade?
    - What was the worst single trade?

    Fields:
        date        : The date this trade happened
        action      : 'BUY' or 'SELL'
        price       : The price at which the trade was executed (after slippage)
        shares      : Number of shares bought or sold
        value       : Total money exchanged (price × shares)
        fee         : Brokerage fee paid on this trade
        signal_price: The raw closing price on the signal day (before slippage)
    """
    date:         pd.Timestamp
    action:       str         # 'BUY' or 'SELL'
    price:        float       # execution price (after slippage)
    shares:       int         # number of shares
    value:        float       # price × shares
    fee:          float       # brokerage fee
    signal_price: float       # close price on signal day (before slippage)


@dataclass
class PortfolioSnapshot:
    """
    A snapshot of the portfolio's state on one specific day.
    We record one of these for every trading day — this builds the equity curve.

    Fields:
        date    : The trading day
        cash    : Cash available (not invested)
        shares  : Shares held
        price   : Today's closing price of the stock
        equity  : cash + (shares × price) — total portfolio value
        signal  : The signal on this day (+1, -1, 0)
    """
    date:   pd.Timestamp
    cash:   float
    shares: int
    price:  float
    equity: float
    signal: int


# =============================================================================
# LAYER 3: EXECUTION SIMULATOR
# =============================================================================
# Converts raw BUY/SELL signals into realistic trade executions.
#
# The realism gap:
#   Your strategy generates a signal at the END of day D (after close).
#   You can only act on it at the START of day D+1 (at open).
#   So: signal on day D → trade executes at day D+1's OPEN price.
#
# This is called "next-day open execution" and it's the most realistic
# assumption for a daily backtesting engine without HFT infrastructure.

class ExecutionSimulator:
    """
    Simulates how trades are actually executed in the real market.

    Handles:
        - Next-day open execution (signal day ≠ execution day)
        - Brokerage fees (percentage of trade value)
        - Slippage (price moves against you on execution)
        - Position sizing (how many shares to buy with available cash)

    Args:
        brokerage_pct : Fee as % of trade value. Default 0.1% (₹10 per ₹10,000).
                        Typical Indian brokers: Zerodha charges ₹20 flat or 0.03%.
                        We use 0.1% as a conservative estimate.
        slippage_pct  : % by which price moves against you.
                        BUY: you pay slightly MORE than open price.
                        SELL: you receive slightly LESS than open price.
                        Default 0.05% — realistic for liquid large-cap stocks.
        position_pct  : What fraction of available cash to invest per BUY signal.
                        Default 1.0 (100%) — invest all available cash.
                        Set to 0.5 to invest only 50% and keep 50% as reserve.
    """

    def __init__(
        self,
        brokerage_pct: float = 0.001,   # 0.1%
        slippage_pct:  float = 0.0005,  # 0.05%
        position_pct:  float = 1.0,     # 100% of available cash
    ):
        self.brokerage_pct = brokerage_pct
        self.slippage_pct  = slippage_pct
        self.position_pct  = position_pct

    def calculate_buy_price(self, open_price: float) -> float:
        """
        When we BUY, we pay the open price PLUS slippage.
        Slippage means: "by the time our order fills, the price
        has moved up slightly against us."
        Example: open = ₹2500, slippage 0.05% → execution price = ₹2501.25
        """
        return open_price * (1 + self.slippage_pct)

    def calculate_sell_price(self, open_price: float) -> float:
        """
        When we SELL, we receive the open price MINUS slippage.
        Slippage here means: "filling a sell order pushes price down slightly."
        Example: open = ₹2500, slippage 0.05% → execution price = ₹2498.75
        """
        return open_price * (1 - self.slippage_pct)

    def calculate_shares_to_buy(self, cash: float, execution_price: float) -> int:
        """
        Calculates how many WHOLE shares we can buy.

        Why whole shares?
            In real markets you cannot buy 2.7 shares. It must be a whole number.
            We use int() to round DOWN (never exceed available cash).

        Args:
            cash            : Available cash in the portfolio
            execution_price : The price we'll pay per share (after slippage)

        Returns:
            Number of whole shares we can afford (0 if not enough cash)
        """
        # Only deploy position_pct fraction of available cash
        deployable_cash = cash * self.position_pct

        # Calculate fee as fraction of the total trade value
        # Since fee = trade_value × brokerage_pct and trade_value = shares × price,
        # effective cost per share = price × (1 + brokerage_pct)
        effective_cost_per_share = execution_price * (1 + self.brokerage_pct)

        shares = int(deployable_cash / effective_cost_per_share)
        return max(shares, 0)   # never negative

    def calculate_fee(self, trade_value: float) -> float:
        """Fee = trade_value × brokerage percentage"""
        return trade_value * self.brokerage_pct

    def execute_buy(
        self,
        date: pd.Timestamp,
        open_price: float,
        signal_price: float,
        cash: float,
    ) -> Trade | None:
        """
        Attempts to execute a BUY order.

        Returns a Trade object if successful, or None if:
            - We don't have enough cash for even 1 share
            - open_price is NaN (data gap)

        Args:
            date         : The execution date (day AFTER signal)
            open_price   : Tomorrow's open price (our execution price basis)
            signal_price : Yesterday's close (the price that triggered the signal)
            cash         : Available cash in portfolio
        """
        if pd.isna(open_price) or open_price <= 0:
            logger.warning(f"  BUY skipped on {date.date()}: invalid open price")
            return None

        exec_price = self.calculate_buy_price(open_price)
        shares     = self.calculate_shares_to_buy(cash, exec_price)

        if shares == 0:
            logger.warning(f"  BUY skipped on {date.date()}: insufficient cash ₹{cash:.2f}")
            return None

        value = shares * exec_price
        fee   = self.calculate_fee(value)

        logger.info(
            f"  BUY  {date.date()} | {shares} shares @ ₹{exec_price:.2f} "
            f"| value=₹{value:.2f} | fee=₹{fee:.2f}"
        )

        return Trade(
            date=date, action='BUY', price=exec_price,
            shares=shares, value=value, fee=fee, signal_price=signal_price
        )

    def execute_sell(
        self,
        date: pd.Timestamp,
        open_price: float,
        signal_price: float,
        shares_held: int,
    ) -> Trade | None:
        """
        Attempts to execute a SELL order.

        Returns a Trade object if successful, or None if:
            - We hold 0 shares (nothing to sell)
            - open_price is invalid
        """
        if pd.isna(open_price) or open_price <= 0:
            logger.warning(f"  SELL skipped on {date.date()}: invalid open price")
            return None

        if shares_held <= 0:
            # This happens when strategy fires SELL but we never bought
            # (e.g. first signal was SELL, or we couldn't afford to buy earlier)
            logger.debug(f"  SELL skipped on {date.date()}: no shares held")
            return None

        exec_price = self.calculate_sell_price(open_price)
        value      = shares_held * exec_price
        fee        = self.calculate_fee(value)

        logger.info(
            f"  SELL {date.date()} | {shares_held} shares @ ₹{exec_price:.2f} "
            f"| value=₹{value:.2f} | fee=₹{fee:.2f}"
        )

        return Trade(
            date=date, action='SELL', price=exec_price,
            shares=shares_held, value=value, fee=fee, signal_price=signal_price
        )


# =============================================================================
# LAYER 4: PORTFOLIO TRACKER
# =============================================================================
# Keeps the running state of the portfolio (cash, shares, equity)
# and updates it on every bar.
#
# The main loop:
#   For each trading day:
#     1. Check if yesterday's signal requires a trade today
#     2. If so, ask ExecutionSimulator to execute it
#     3. Update cash and shares based on the trade result
#     4. Calculate today's equity = cash + (shares × today's close)
#     5. Record a PortfolioSnapshot
#
# The collection of all PortfolioSnapshots = the EQUITY CURVE.

class PortfolioTracker:
    """
    Manages portfolio state and runs the main backtest loop.

    This is the engine that connects:
        Layer 2 output (signals) → Layer 3 (execution) → equity curve

    Args:
        initial_capital : Starting cash in ₹. Default ₹100,000 (1 lakh).
        simulator       : An ExecutionSimulator instance. If None, uses defaults.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        simulator: ExecutionSimulator = None,
    ):
        self.initial_capital = initial_capital
        self.simulator       = simulator or ExecutionSimulator()

        # Current state (updated on every bar)
        self.cash   = initial_capital   # cash available
        self.shares = 0                 # shares currently held

        # History (one entry per trading day)
        self.snapshots: List[PortfolioSnapshot] = []
        self.trades:    List[Trade]             = []

        logger.info(
            f"PortfolioTracker ready | "
            f"Capital: ₹{initial_capital:,.2f} | "
            f"Simulator: brokerage={self.simulator.brokerage_pct*100:.2f}%, "
            f"slippage={self.simulator.slippage_pct*100:.3f}%"
        )

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full backtest loop over the signal DataFrame.

        Args:
            df: DataFrame from Layer 2 — must contain columns:
                [open, close, signal, ema_fast, ema_slow, ...]

        Returns:
            DataFrame with equity curve and trade markers added as new columns:
                equity        : portfolio value on each day
                cash_held     : cash portion of portfolio each day
                position_value: value of shares held each day
                trade_action  : 'BUY', 'SELL', or '' for each day
        """
        # Reset state in case run() is called multiple times
        self.cash    = self.initial_capital
        self.shares  = 0
        self.snapshots = []
        self.trades    = []

        logger.info(f"Starting backtest over {len(df)} bars...")
        logger.info(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")

        rows = df.reset_index()  # Convert DatetimeIndex to a column for iteration

        for i, row in rows.iterrows():
            date       = row['date'] if 'date' in row else df.index[i]
            signal     = int(row['signal'])
            open_price = row['open']
            close      = row['close']

            trade_executed = None

            # --- PROCESS SIGNAL -------------------------------------------
            # Signal was generated YESTERDAY at market close.
            # We act on it TODAY at market OPEN.
            # (This is already handled because Layer 2 shifts indicators by 1.)

            if signal == 1 and self.shares == 0:
                # BUY signal AND we don't already hold a position
                # (We only hold one position at a time — "long only" strategy)
                trade = self.simulator.execute_buy(
                    date=date,
                    open_price=open_price,
                    signal_price=row['close'],
                    cash=self.cash,
                )
                if trade:
                    # Deduct cost from cash
                    self.cash   -= (trade.value + trade.fee)
                    self.shares  = trade.shares
                    self.trades.append(trade)
                    trade_executed = trade

            elif signal == -1 and self.shares > 0:
                # SELL signal AND we actually hold shares
                trade = self.simulator.execute_sell(
                    date=date,
                    open_price=open_price,
                    signal_price=row['close'],
                    shares_held=self.shares,
                )
                if trade:
                    # Add proceeds to cash (minus fee)
                    self.cash  += (trade.value - trade.fee)
                    self.shares = 0
                    self.trades.append(trade)
                    trade_executed = trade

            # --- RECORD DAILY SNAPSHOT ------------------------------------
            # Equity = cash + market value of shares held
            # Even on days we don't trade, equity changes as price moves
            position_value = self.shares * close
            equity         = self.cash + position_value

            self.snapshots.append(PortfolioSnapshot(
                date=date,
                cash=self.cash,
                shares=self.shares,
                price=close,
                equity=equity,
                signal=signal,
            ))

        # --- BUILD RESULTS DATAFRAME --------------------------------------
        results = self._build_results_df(df)

        total_return = (results['equity'].iloc[-1] / self.initial_capital - 1) * 100
        total_trades = len(self.trades)
        logger.info(
            f"Backtest complete | "
            f"Trades: {total_trades} | "
            f"Final equity: ₹{results['equity'].iloc[-1]:,.2f} | "
            f"Total return: {total_return:+.2f}%"
        )

        return results

    def _build_results_df(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines the original signal DataFrame with the equity curve.
        Returns a single DataFrame with everything in one place.
        """
        equity_data = {
            'equity':         [s.equity         for s in self.snapshots],
            'cash_held':      [s.cash           for s in self.snapshots],
            'position_value': [s.shares * s.price for s in self.snapshots],
        }
        equity_df = pd.DataFrame(equity_data, index=original_df.index)

        # Add trade action markers (makes it easy to plot trade points)
        trade_map = {t.date: t.action for t in self.trades}
        equity_df['trade_action'] = [
            trade_map.get(date, '') for date in original_df.index
        ]

        # Combine with original signals DataFrame
        result = pd.concat([original_df, equity_df], axis=1)
        return result

    def get_trade_log(self) -> pd.DataFrame:
        """
        Returns all trades as a clean DataFrame.
        Useful for analysing individual trade performance.
        """
        if not self.trades:
            return pd.DataFrame()

        data = [{
            'date':         t.date,
            'action':       t.action,
            'price':        round(t.price, 2),
            'shares':       t.shares,
            'value':        round(t.value, 2),
            'fee':          round(t.fee, 2),
            'signal_price': round(t.signal_price, 2),
        } for t in self.trades]

        return pd.DataFrame(data).set_index('date')

    def get_summary(self) -> dict:
        """
        Returns a dictionary of key portfolio statistics.
        These will be used by Layer 5 (Performance Metrics) for deeper analysis.
        """
        if not self.snapshots:
            return {}

        equity_series = pd.Series(
            [s.equity for s in self.snapshots],
            index=[s.date for s in self.snapshots]
        )

        final_equity   = equity_series.iloc[-1]
        total_return   = (final_equity / self.initial_capital - 1) * 100
        n_trades       = len(self.trades)
        n_buys         = sum(1 for t in self.trades if t.action == 'BUY')
        n_sells        = sum(1 for t in self.trades if t.action == 'SELL')
        total_fees     = sum(t.fee for t in self.trades)

        return {
            'initial_capital': self.initial_capital,
            'final_equity':    round(final_equity, 2),
            'total_return_pct':round(total_return, 2),
            'n_trades':        n_trades,
            'n_buys':          n_buys,
            'n_sells':         n_sells,
            'total_fees_paid': round(total_fees, 2),
        }


# =============================================================================
# DEMO: Run this file directly to test Layers 3 + 4 on real data
# =============================================================================
if __name__ == "__main__":
    from data_loader import DataLoader
    from strategy_engine import EMACrossoverStrategy, EMARSIStrategy

    print("\n" + "="*60)
    print("  BACKTESTING ENGINE — LAYER 3 + 4 TEST")
    print("="*60 + "\n")

    # --- Step 1: Load data (Layer 1) ----------------------------------------
    loader = DataLoader()
    df     = loader.get("RELIANCE.NS", "2021-01-01", "2024-01-01")
    print(f"Loaded {len(df)} trading days of RELIANCE.NS\n")

    # --- Step 2: Generate signals (Layer 2) ---------------------------------
    strategy = EMACrossoverStrategy(fast_period=9, slow_period=21)
    df_signals = strategy.generate_signals(df)

    # --- Step 3 + 4: Execution + Portfolio ----------------------------------
    print("TEST 1: EMA Crossover | ₹1,00,000 starting capital")
    print("-" * 50)

    simulator = ExecutionSimulator(
        brokerage_pct=0.001,   # 0.1% brokerage (typical retail broker)
        slippage_pct=0.0005,   # 0.05% slippage
        position_pct=1.0,      # invest all available cash on each buy
    )
    portfolio = PortfolioTracker(initial_capital=100_000, simulator=simulator)
    results   = portfolio.run(df_signals)

    # --- Show trade log -----------------------------------------------------
    print("\nTrade Log:")
    print(portfolio.get_trade_log().to_string())

    # --- Show summary -------------------------------------------------------
    print("\nPortfolio Summary:")
    summary = portfolio.get_summary()
    for key, val in summary.items():
        print(f"  {key:<22}: {val}")

    # --- Show equity curve (first and last 5 rows) -------------------------
    print("\nEquity Curve (first 5 days):")
    print(results[['close', 'signal', 'equity', 'cash_held',
                   'position_value', 'trade_action']].head().round(2))

    print("\nEquity Curve (last 5 days):")
    print(results[['close', 'signal', 'equity', 'cash_held',
                   'position_value', 'trade_action']].tail().round(2))

    # --- Test 2: EMA + RSI Strategy ----------------------------------------
    print("\n" + "-"*50)
    print("TEST 2: EMA + RSI Strategy | ₹1,00,000 starting capital")
    print("-" * 50)

    strategy2  = EMARSIStrategy(fast_period=9, slow_period=21, rsi_period=14)
    df_sig2    = strategy2.generate_signals(df)
    portfolio2 = PortfolioTracker(initial_capital=100_000, simulator=simulator)
    results2   = portfolio2.run(df_sig2)

    summary2 = portfolio2.get_summary()
    print("\nPortfolio Summary:")
    for key, val in summary2.items():
        print(f"  {key:<22}: {val}")

    # --- Compare both strategies -------------------------------------------
    print("\n" + "="*60)
    print("  STRATEGY COMPARISON")
    print("="*60)
    print(f"  {'Strategy':<25} {'Return':>8} {'Trades':>8} {'Fees':>10}")
    print(f"  {'-'*55}")
    for name, s in [("EMA Crossover", summary), ("EMA + RSI", summary2)]:
        print(
            f"  {name:<25} "
            f"{s['total_return_pct']:>+7.2f}% "
            f"{s['n_trades']:>8} "
            f"₹{s['total_fees_paid']:>8.2f}"
        )
    print("="*60 + "\n")