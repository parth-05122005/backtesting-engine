# =============================================================================
# BACKTESTING ENGINE — LAYER 5: PERFORMANCE METRICS + VISUALISATION
# =============================================================================
# What is this file?
#   The final layer. It takes the equity curve produced by Layer 4 and
#   answers the most important question: "Was this strategy actually good?"
#
# Why can't we just look at total return?
#   A strategy that made +20% by making ONE giant risky bet is very different
#   from a strategy that made +20% steadily with small, controlled moves.
#   Performance metrics capture the QUALITY of returns, not just the quantity.
#
# What metrics do we calculate?
#   1. Total Return     — how much money did we make overall?
#   2. CAGR             — what was our annualised growth rate?
#   3. Sharpe Ratio     — how much return did we earn per unit of risk?
#   4. Max Drawdown     — what was the worst peak-to-trough loss?
#   5. Win Rate         — what % of trades were profitable?
#   6. Profit Factor    — total profit / total loss across all trades
#   7. Avg Trade Return — average P&L per trade
#
# Why implement these from scratch instead of using a library?
#   Quadeye and other quant firms WILL ask you to derive these formulas.
#   If you just call pyfolio.sharpe_ratio(), you can't explain what it does.
#   Building from scratch means you truly understand each metric.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PERFORMANCE METRICS — all built from scratch
# =============================================================================

class PerformanceMetrics:
    """
    Calculates all key performance metrics from an equity curve.

    All metrics are implemented from mathematical first principles.
    No external libraries (pyfolio, quantstats, etc.) used.

    Args:
        results         : The DataFrame returned by PortfolioTracker.run()
                          Must contain: ['equity', 'close', 'trade_action']
        trades_df       : The DataFrame returned by PortfolioTracker.get_trade_log()
        initial_capital : Starting portfolio value
        trading_days    : Number of trading days per year. Default 252 (NSE standard)
    """

    def __init__(
        self,
        results:         pd.DataFrame,
        trades_df:       pd.DataFrame,
        initial_capital: float = 100_000.0,
        trading_days:    int   = 252,
    ):
        self.results         = results
        self.trades_df       = trades_df
        self.initial_capital = initial_capital
        self.trading_days    = trading_days

        # Extract the equity series — the core input to all metrics
        self.equity = results['equity']

        # Daily returns = percentage change in equity from one day to the next
        # Formula: r_t = (equity_t - equity_{t-1}) / equity_{t-1}
        # Why percentage and not absolute?
        #   Because ₹1000 gain on a ₹10,000 portfolio (10%) is very different
        #   from ₹1000 gain on a ₹1,00,000 portfolio (1%).
        self.daily_returns = self.equity.pct_change().dropna()

        logger.info("PerformanceMetrics initialised.")

    # -------------------------------------------------------------------------
    # METRIC 1: Total Return
    # -------------------------------------------------------------------------
    def total_return(self) -> float:
        """
        Total Return = (Final Value - Initial Value) / Initial Value × 100

        The simplest metric. How much did our ₹ grow in percentage terms?

        Example: Started with ₹1,00,000. Ended with ₹1,20,000.
                 Total Return = (1,20,000 - 1,00,000) / 1,00,000 × 100 = +20%
        """
        final = self.equity.iloc[-1]
        return (final - self.initial_capital) / self.initial_capital * 100

    # -------------------------------------------------------------------------
    # METRIC 2: CAGR — Compound Annual Growth Rate
    # -------------------------------------------------------------------------
    def cagr(self) -> float:
        """
        CAGR = (Final Value / Initial Value) ^ (1 / years) - 1

        Why CAGR instead of total return?
            Total return doesn't account for HOW LONG you held the investment.
            +20% over 1 year is amazing. +20% over 10 years is terrible.
            CAGR normalises return to a per-year equivalent.

        Example: ₹1,00,000 → ₹1,46,000 over 3 years
                 CAGR = (1.46)^(1/3) - 1 = 13.4% per year

        The ^(1/n) is the nth root — it "undoes" n years of compounding
        to find what constant annual rate would produce the same result.
        """
        n_days = len(self.equity)
        years  = n_days / self.trading_days

        if years <= 0:
            return 0.0

        final   = self.equity.iloc[-1]
        initial = self.equity.iloc[0]

        return ((final / initial) ** (1 / years) - 1) * 100

    # -------------------------------------------------------------------------
    # METRIC 3: Sharpe Ratio
    # -------------------------------------------------------------------------
    def sharpe_ratio(self, risk_free_rate: float = 0.065) -> float:
        """
        Sharpe Ratio = (Mean Daily Return - Risk-Free Daily Return) / Std of Daily Returns
                       × sqrt(252)   ← annualise it

        What does it measure?
            Return earned PER UNIT OF RISK taken.
            A Sharpe of 1.0 means: for every 1% of volatility you accept,
            you earn 1% of excess return (above risk-free rate).

        Interpretation:
            < 0.5  : Poor — not worth the risk
            0.5–1  : Below average
            1–1.5  : Good
            1.5–2  : Very good
            > 2    : Excellent (rare in practice)

        Why subtract the risk-free rate?
            Risk-free rate = what you'd earn by doing NOTHING (e.g. putting money
            in an FD or government bond). In India, this is ~6.5% (RBI repo rate).
            We only want to measure the EXTRA return for taking equity risk.

        Why multiply by sqrt(252)?
            We compute daily Sharpe. To get annual Sharpe, multiply by sqrt(252).
            This comes from the fact that variance scales linearly with time,
            so standard deviation scales with the square root of time.
            (This is a standard result from statistics.)

        Args:
            risk_free_rate: Annual risk-free rate. Default 6.5% (Indian T-bill).
        """
        if self.daily_returns.std() == 0:
            return 0.0

        # Convert annual risk-free rate to daily equivalent
        # (1 + annual_rate)^(1/252) - 1 ≈ annual_rate / 252 for small rates
        daily_rf = risk_free_rate / self.trading_days

        excess_returns = self.daily_returns - daily_rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.trading_days)
        return round(sharpe, 4)

    # -------------------------------------------------------------------------
    # METRIC 4: Maximum Drawdown
    # -------------------------------------------------------------------------
    def max_drawdown(self) -> float:
        """
        Max Drawdown = worst peak-to-trough decline in portfolio value

        Formula:
            running_peak[t] = max(equity[0], equity[1], ..., equity[t])
            drawdown[t]     = (equity[t] - running_peak[t]) / running_peak[t]
            max_drawdown    = min(drawdown)  ← most negative value

        What does it measure?
            The largest single "bad streak" your portfolio experienced.
            If your portfolio went: ₹1L → ₹1.3L → ₹0.9L → ₹1.1L
            The peak was ₹1.3L, the trough was ₹0.9L.
            Max Drawdown = (0.9 - 1.3) / 1.3 = -30.8%

        Why does this matter?
            Psychologically, a -30% drawdown is extremely hard to hold through.
            Many investors panic and sell at the bottom.
            A strategy with -5% max drawdown is much more "tradeable" than
            one with -40% max drawdown, even if both have the same total return.

        Returns:
            A negative percentage (e.g. -15.3 means -15.3% max drawdown)
        """
        # Running peak: on each day, what was the highest equity ever seen?
        running_peak = self.equity.cummax()

        # Drawdown on each day: how far below the peak are we?
        drawdown = (self.equity - running_peak) / running_peak * 100

        return round(drawdown.min(), 4)   # most negative = worst drawdown

    def drawdown_series(self) -> pd.Series:
        """Returns the full drawdown series (one value per day) for plotting."""
        running_peak = self.equity.cummax()
        return (self.equity - running_peak) / running_peak * 100

    # -------------------------------------------------------------------------
    # METRIC 5: Win Rate
    # -------------------------------------------------------------------------
    def win_rate(self) -> float:
        """
        Win Rate = (Number of profitable trades / Total closed trades) × 100

        A "closed trade" = one BUY matched with the subsequent SELL.
        We match them in pairs: [BUY₁, SELL₁], [BUY₂, SELL₂], ...

        Example:
            Trade 1: Bought @ ₹2400, Sold @ ₹2600 → +₹200 profit → WIN
            Trade 2: Bought @ ₹2800, Sold @ ₹2650 → -₹150 loss   → LOSS
            Trade 3: Bought @ ₹2500, Sold @ ₹2700 → +₹200 profit → WIN
            Win rate = 2/3 = 66.7%

        Returns 0 if no completed round-trips exist.
        """
        if self.trades_df.empty:
            return 0.0

        buys  = self.trades_df[self.trades_df['action'] == 'BUY'].reset_index()
        sells = self.trades_df[self.trades_df['action'] == 'SELL'].reset_index()

        n_pairs = min(len(buys), len(sells))
        if n_pairs == 0:
            return 0.0

        wins = 0
        for i in range(n_pairs):
            buy_price  = buys.iloc[i]['price']
            sell_price = sells.iloc[i]['price']
            if sell_price > buy_price:
                wins += 1

        return round(wins / n_pairs * 100, 2)

    # -------------------------------------------------------------------------
    # METRIC 6: Profit Factor
    # -------------------------------------------------------------------------
    def profit_factor(self) -> float:
        """
        Profit Factor = Total Gross Profit / Total Gross Loss

        Measures how much you win vs how much you lose in absolute terms.

        Interpretation:
            < 1.0  : Losing strategy (losses exceed profits)
            1.0    : Break-even
            1.0–1.5: Marginally profitable
            1.5–2.0: Good
            > 2.0  : Excellent

        Example:
            3 winning trades: +₹500, +₹300, +₹700 → gross profit = ₹1500
            2 losing trades: -₹200, -₹400          → gross loss   = ₹600
            Profit Factor = 1500 / 600 = 2.5 → excellent!
        """
        if self.trades_df.empty:
            return 0.0

        buys  = self.trades_df[self.trades_df['action'] == 'BUY'].reset_index()
        sells = self.trades_df[self.trades_df['action'] == 'SELL'].reset_index()

        n_pairs = min(len(buys), len(sells))
        if n_pairs == 0:
            return 0.0

        gross_profit = 0.0
        gross_loss   = 0.0

        for i in range(n_pairs):
            pnl = (sells.iloc[i]['price'] - buys.iloc[i]['price']) * buys.iloc[i]['shares']
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss   += abs(pnl)

        if gross_loss == 0:
            return float('inf')   # all trades were winners

        return round(gross_profit / gross_loss, 4)

    # -------------------------------------------------------------------------
    # METRIC 7: Average Trade Return
    # -------------------------------------------------------------------------
    def avg_trade_return(self) -> float:
        """
        Average Trade Return = mean of (sell_price - buy_price) / buy_price
        across all completed round-trips, expressed as a percentage.

        Tells you: on average, how much % did you make per trade?
        A high win rate with very small avg returns is worse than a lower
        win rate with large avg returns. Both matter together.
        """
        if self.trades_df.empty:
            return 0.0

        buys  = self.trades_df[self.trades_df['action'] == 'BUY'].reset_index()
        sells = self.trades_df[self.trades_df['action'] == 'SELL'].reset_index()

        n_pairs = min(len(buys), len(sells))
        if n_pairs == 0:
            return 0.0

        returns = []
        for i in range(n_pairs):
            r = (sells.iloc[i]['price'] - buys.iloc[i]['price']) / buys.iloc[i]['price'] * 100
            returns.append(r)

        return round(np.mean(returns), 4)

    # -------------------------------------------------------------------------
    # COMPUTE ALL METRICS AT ONCE
    # -------------------------------------------------------------------------
    def compute_all(self) -> dict:
        """
        Computes and returns all metrics as a clean dictionary.
        This is what you'd log, save, or pass to a dashboard.
        """
        metrics = {
            'Total Return (%)':    round(self.total_return(), 2),
            'CAGR (%)':            round(self.cagr(), 2),
            'Sharpe Ratio':        self.sharpe_ratio(),
            'Max Drawdown (%)':    self.max_drawdown(),
            'Win Rate (%)':        self.win_rate(),
            'Profit Factor':       self.profit_factor(),
            'Avg Trade Return (%)':self.avg_trade_return(),
            'Total Trades':        len(self.trades_df),
        }
        return metrics

    def print_report(self, strategy_name: str = "Strategy"):
        """Prints a formatted performance report to the console."""
        metrics = self.compute_all()

        print(f"\n{'='*55}")
        print(f"  PERFORMANCE REPORT: {strategy_name}")
        print(f"{'='*55}")
        for name, val in metrics.items():
            # Colour-code returns and Sharpe in the console
            if isinstance(val, float):
                print(f"  {name:<26}: {val:>+.2f}" if '%' in name or name == 'Sharpe Ratio'
                      else f"  {name:<26}: {val:>.4f}")
            else:
                print(f"  {name:<26}: {val}")
        print(f"{'='*55}\n")


# =============================================================================
# VISUALISATION
# =============================================================================

class BacktestVisualiser:
    """
    Plots the equity curve, drawdown, trade markers, and buy-and-hold benchmark.

    Produces a 3-panel chart:
        Panel 1 (top)   : Price chart with EMA lines + BUY/SELL markers
        Panel 2 (middle): Equity curve vs buy-and-hold benchmark
        Panel 3 (bottom): Drawdown chart
    """

    def __init__(self, results: pd.DataFrame, initial_capital: float = 100_000):
        self.results         = results
        self.initial_capital = initial_capital

    def plot(self, strategy_name: str = "Strategy", save_path: str = None):
        """
        Generates and shows the full backtest chart.

        Args:
            strategy_name: Name shown in chart title
            save_path    : If provided, saves chart to this file path
                           e.g. "results/RELIANCE_backtest.png"
        """
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#0d1117')   # dark background

        # GridSpec: 3 rows with different heights
        # Panel 1 takes 40%, Panel 2 takes 40%, Panel 3 takes 20%
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.08)

        ax1 = fig.add_subplot(gs[0])   # Price + EMA + signals
        ax2 = fig.add_subplot(gs[1])   # Equity curve
        ax3 = fig.add_subplot(gs[2])   # Drawdown

        # Shared x-axis formatting
        for ax in [ax1, ax2]:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e', labelbottom=False)
            ax.spines[:].set_color('#30363d')
            ax.yaxis.label.set_color('#8b949e')
            ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--')

        ax3.set_facecolor('#0d1117')
        ax3.tick_params(colors='#8b949e')
        ax3.spines[:].set_color('#30363d')
        ax3.yaxis.label.set_color('#8b949e')
        ax3.grid(True, color='#21262d', linewidth=0.5, linestyle='--')

        dates = self.results.index

        # ── PANEL 1: Price chart ─────────────────────────────────────────────
        ax1.plot(dates, self.results['close'],
                 color='#58a6ff', linewidth=1.2, label='Close price', zorder=2)

        if 'ema_fast' in self.results.columns:
            ax1.plot(dates, self.results['ema_fast'],
                     color='#f0883e', linewidth=0.9, linestyle='--',
                     label='EMA Fast', zorder=2)

        if 'ema_slow' in self.results.columns:
            ax1.plot(dates, self.results['ema_slow'],
                     color='#d2a8ff', linewidth=0.9, linestyle='--',
                     label='EMA Slow', zorder=2)

        # BUY markers: green upward triangles
        buys  = self.results[self.results['trade_action'] == 'BUY']
        sells = self.results[self.results['trade_action'] == 'SELL']

        ax1.scatter(buys.index, buys['close'],
                    marker='^', color='#3fb950', s=80, zorder=5,
                    label='BUY signal')
        ax1.scatter(sells.index, sells['close'],
                    marker='v', color='#f85149', s=80, zorder=5,
                    label='SELL signal')

        ax1.set_ylabel('Price (₹)', color='#8b949e')
        ax1.legend(loc='upper left', fontsize=8,
                   facecolor='#161b22', labelcolor='#c9d1d9', framealpha=0.8)
        ax1.set_title(
            f'{strategy_name} — Backtest Results',
            color='#c9d1d9', fontsize=13, pad=12
        )

        # ── PANEL 2: Equity curve ────────────────────────────────────────────
        # Buy-and-hold benchmark: what if we just bought on day 1 and held?
        bah_equity = (self.results['close'] / self.results['close'].iloc[0]) * self.initial_capital

        ax2.plot(dates, self.results['equity'],
                 color='#3fb950', linewidth=1.5, label='Strategy equity', zorder=3)
        ax2.plot(dates, bah_equity,
                 color='#58a6ff', linewidth=1.2, linestyle='--',
                 label='Buy & Hold', zorder=2, alpha=0.7)

        # Shade the area between strategy and benchmark
        ax2.fill_between(
            dates,
            self.results['equity'], bah_equity,
            where=self.results['equity'] >= bah_equity,
            alpha=0.1, color='#3fb950', label='Outperforming'
        )
        ax2.fill_between(
            dates,
            self.results['equity'], bah_equity,
            where=self.results['equity'] < bah_equity,
            alpha=0.1, color='#f85149', label='Underperforming'
        )

        ax2.axhline(y=self.initial_capital, color='#8b949e',
                    linewidth=0.7, linestyle=':', alpha=0.5)
        ax2.set_ylabel('Portfolio Value (₹)', color='#8b949e')
        ax2.legend(loc='upper left', fontsize=8,
                   facecolor='#161b22', labelcolor='#c9d1d9', framealpha=0.8)

        # Format y-axis in lakhs for readability
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K')
        )

        # ── PANEL 3: Drawdown ────────────────────────────────────────────────
        dd = (self.results['equity'] / self.results['equity'].cummax() - 1) * 100
        ax3.fill_between(dates, dd, 0, color='#f85149', alpha=0.4, label='Drawdown')
        ax3.plot(dates, dd, color='#f85149', linewidth=0.8)
        ax3.set_ylabel('Drawdown (%)', color='#8b949e')
        ax3.set_xlabel('Date', color='#8b949e')

        # Format date axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            logger.info(f"Chart saved to: {save_path}")

        plt.show()
        return fig


# =============================================================================
# FULL BACKTEST RUNNER — ties all 5 layers together
# =============================================================================

class Backtester:
    """
    The master class that wires all 5 layers together into one clean interface.

    Usage:
        bt = Backtester(
            ticker="RELIANCE.NS",
            start="2021-01-01",
            end="2024-01-01",
            initial_capital=100_000
        )
        bt.run(strategy=EMACrossoverStrategy(9, 21))
        bt.report()
        bt.plot()
    """

    def __init__(
        self,
        ticker:          str,
        start:           str,
        end:             str,
        initial_capital: float = 100_000.0,
    ):
        self.ticker          = ticker
        self.start           = start
        self.end             = end
        self.initial_capital = initial_capital

        self.results   = None
        self.trades_df = None
        self.metrics   = None

    def run(self, strategy, simulator=None):
        """
        Runs the full pipeline: fetch → signals → execute → portfolio → metrics.

        Args:
            strategy : Any Strategy subclass instance (from Layer 2)
            simulator: An ExecutionSimulator instance (optional, uses defaults)
        """
        from data_loader     import DataLoader
        from portfolio       import ExecutionSimulator, PortfolioTracker

        logger.info(f"\n{'─'*55}")
        logger.info(f"  Backtest: {self.ticker} | {self.start} → {self.end}")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"{'─'*55}")

        # Layer 1: Data
        loader = DataLoader()
        df     = loader.get(self.ticker, self.start, self.end)

        # Layer 2: Signals
        df_signals = strategy.generate_signals(df)

        # Layer 3 + 4: Execution + Portfolio
        sim       = simulator or ExecutionSimulator()
        portfolio = PortfolioTracker(self.initial_capital, sim)
        self.results   = portfolio.run(df_signals)
        self.trades_df = portfolio.get_trade_log()

        # Layer 5: Metrics
        self.metrics = PerformanceMetrics(
            results=self.results,
            trades_df=self.trades_df,
            initial_capital=self.initial_capital,
        )
        return self

    def report(self, strategy_name: str = "Strategy"):
        """Prints the full performance report."""
        if self.metrics is None:
            print("Run backtest first: bt.run(strategy)")
            return
        self.metrics.print_report(strategy_name)

    def plot(self, strategy_name: str = "Strategy", save_path: str = None):
        """Plots the full backtest chart."""
        if self.results is None:
            print("Run backtest first: bt.run(strategy)")
            return
        vis = BacktestVisualiser(self.results, self.initial_capital)
        vis.plot(strategy_name, save_path)


# =============================================================================
# DEMO: Run this file directly — full end-to-end test of all 5 layers
# =============================================================================
if __name__ == "__main__":
    from strategy_engine import EMACrossoverStrategy, EMARSIStrategy

    print("\n" + "="*55)
    print("  BACKTESTING ENGINE — FULL END-TO-END TEST")
    print("="*55)

    # ── TEST 1: EMA Crossover on RELIANCE ───────────────────────────────────
    bt1 = Backtester(
        ticker="RELIANCE.NS",
        start="2021-01-01",
        end="2024-01-01",
        initial_capital=100_000,
    )
    bt1.run(strategy=EMACrossoverStrategy(fast_period=9, slow_period=21))
    bt1.report("EMA Crossover (9,21)")
    bt1.plot("RELIANCE.NS — EMA Crossover (9,21)")

    # ── TEST 2: EMA + RSI on RELIANCE ───────────────────────────────────────
    bt2 = Backtester(
        ticker="RELIANCE.NS",
        start="2021-01-01",
        end="2024-01-01",
        initial_capital=100_000,
    )
    bt2.run(strategy=EMARSIStrategy(fast_period=9, slow_period=21))
    bt2.report("EMA + RSI Strategy")
    bt2.plot("RELIANCE.NS — EMA + RSI Strategy")

    # ── TEST 3: Try a different stock ────────────────────────────────────────
    bt3 = Backtester(
        ticker="TCS.NS",
        start="2021-01-01",
        end="2024-01-01",
        initial_capital=100_000,
    )
    bt3.run(strategy=EMACrossoverStrategy(fast_period=9, slow_period=21))
    bt3.report("TCS.NS — EMA Crossover")

    # ── COMPARE all three ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"  {'Strategy':<30} {'Return':>8} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print(f"  {'-'*63}")

    tests = [
        ("RELIANCE EMA Cross", bt1),
        ("RELIANCE EMA+RSI",   bt2),
        ("TCS EMA Cross",      bt3),
    ]

    for name, bt in tests:
        m = bt.metrics.compute_all()
        print(
            f"  {name:<30}"
            f"  {m['Total Return (%)']:>+6.1f}%"
            f"  {m['CAGR (%)']:>+6.1f}%"
            f"  {m['Sharpe Ratio']:>7.2f}"
            f"  {m['Max Drawdown (%)']:>+7.1f}%"
            f"  {m['Win Rate (%)']:>6.1f}%"
        )

    print(f"{'='*65}\n")