# 📈 Backtesting Engine — Indian Stock Market (NSE)

A production-grade backtesting engine built from scratch in Python for NSE-listed Indian stocks. Implements a full 5-layer pipeline from raw data ingestion to performance analytics — with zero dependency on third-party backtesting libraries.

> Built as a quantitative finance portfolio project, demonstrating end-to-end understanding of trading systems, statistical performance metrics, and software architecture.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 BACKTESTING ENGINE                  │
│                                                     │
│  Layer 1        Layer 2        Layer 3 + 4          │
│  Data           Strategy       Execution +          │
│  Ingestion  →   Engine     →   Portfolio        →   │
│                                Tracker              │
│  yfinance       EMA, RSI,      Slippage,            │
│  NSE/.NS        MACD signals   fees, equity     →   │
│  CSV cache      +1 / -1 / 0    curve                │
│                                                     │
│                              Layer 5                │
│                              Performance            │
│                              Metrics +              │
│                              Visualisation          │
│                              Sharpe, CAGR,          │
│                              Drawdown, plots        │
└─────────────────────────────────────────────────────┘
```

Each layer is independently testable and designed to be swappable — you can plug in any strategy without touching the execution or metrics code.

---

## ✨ Features

**Data Layer**
- Fetches historical OHLCV data for any NSE stock via `yfinance` (`.NS` suffix)
- Automatic split & dividend adjustment (`auto_adjust=True`)
- Full data cleaning pipeline: NaN removal, duplicate detection, zero-price filtering, date sorting
- Local CSV cache — subsequent runs load from disk in milliseconds

**Strategy Engine**
- Pluggable `Strategy` base class — implement any strategy in one method
- EMA Crossover Strategy (Golden Cross / Death Cross)
- EMA + RSI Filtered Strategy (reduces false signals in overbought conditions)
- MACD indicator utility included
- Look-ahead bias prevention via `.shift(1)` on all indicators

**Execution Simulator**
- Next-day open price execution (realistic — you can't trade on the signal candle)
- Configurable slippage (default 0.05%)
- Configurable brokerage fees (default 0.1% per trade)
- Position sizing: whole-share calculation with cash constraints

**Portfolio Tracker**
- Tracks cash, shares held, and equity on every bar
- Long-only, one-position-at-a-time model
- Full trade log with date, price, shares, value, fees
- Daily portfolio snapshots for equity curve construction

**Performance Metrics** (all implemented from mathematical first principles)
- Total Return
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio (annualised, with configurable risk-free rate)
- Maximum Drawdown
- Win Rate
- Profit Factor
- Average Trade Return

**Visualisation**
- 3-panel dark-theme chart: price + EMA lines, equity curve vs buy-and-hold, drawdown
- BUY/SELL trade markers on price chart
- Outperformance / underperformance shading vs benchmark

---

## 📁 Project Structure

```
BacktestingEngine/
│
├── data_loader.py        # Layer 1: Data ingestion & caching
├── strategy_engine.py    # Layer 2: Indicators & signal generation
├── portfolio.py          # Layer 3+4: Execution simulator & portfolio tracker
├── metrics.py            # Layer 5: Performance metrics & visualisation
│
├── data/
│   └── raw/              # Auto-created. Cached CSV files stored here.
│
├── results/              # Optional. Save chart outputs here.
│
└── README.md
```

---

## ⚡ Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/parth-05122005/backtesting-engine.git
cd backtesting-engine
```

### 2. Install dependencies
```bash
pip install yfinance pandas numpy matplotlib
```

### 3. Run the full backtest
```bash
python metrics.py
```

This will:
- Download 3 years of RELIANCE.NS and TCS.NS data (cached after first run)
- Run EMA Crossover and EMA+RSI strategies on both stocks
- Print a performance report for each
- Display the 3-panel backtest chart

---

## 🚀 Usage

### Run a single backtest
```python
from strategy_engine import EMACrossoverStrategy
from metrics import Backtester

bt = Backtester(
    ticker="RELIANCE.NS",
    start="2021-01-01",
    end="2024-01-01",
    initial_capital=100_000,
)

bt.run(strategy=EMACrossoverStrategy(fast_period=9, slow_period=21))
bt.report("EMA Crossover (9, 21)")
bt.plot("RELIANCE.NS — EMA Crossover")
```

### Compare multiple strategies
```python
from strategy_engine import EMACrossoverStrategy, EMARSIStrategy
from metrics import Backtester

strategies = {
    "EMA Cross (9,21)":  EMACrossoverStrategy(9, 21),
    "EMA Cross (5,13)":  EMACrossoverStrategy(5, 13),
    "EMA + RSI":         EMARSIStrategy(9, 21, rsi_period=14, rsi_overbought=70),
}

for name, strategy in strategies.items():
    bt = Backtester("INFY.NS", "2021-01-01", "2024-01-01", 100_000)
    bt.run(strategy)
    bt.report(name)
```

### Add your own strategy
```python
from strategy_engine import Strategy
import pandas as pd

class MyStrategy(Strategy):
    """
    Example: Simple RSI mean-reversion strategy.
    Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought).
    """
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        from strategy_engine import rsi
        df = df.copy()
        df['rsi'] = rsi(df['close'], period=14).shift(1)  # shift: no look-ahead bias
        df['signal'] = 0
        df.loc[df['rsi'] < 30, 'signal'] =  1   # BUY
        df.loc[df['rsi'] > 70, 'signal'] = -1   # SELL
        return df
```

### Supported NSE tickers
```
RELIANCE.NS   TCS.NS        INFY.NS       HDFCBANK.NS
ICICIBANK.NS  SBIN.NS       WIPRO.NS      BAJFINANCE.NS
HINDUNILVR.NS AXISBANK.NS   ^NSEI (Nifty 50)
```
Any NSE symbol + `.NS` suffix works. Find tickers at [finance.yahoo.com](https://finance.yahoo.com).


## 🧠 Key Concepts Implemented

### Look-Ahead Bias Prevention
The most common backtesting mistake. Every indicator is shifted by one bar before signal generation:
```python
# WRONG — uses today's close to generate today's signal (impossible in real trading)
df['signal'] = df['close'].rolling(20).mean()

# CORRECT — today's signal only uses data available yesterday
df['signal'] = df['close'].rolling(20).mean().shift(1)
```

### Realistic Execution Modelling
Signals fire at end of day D. Trades execute at open of day D+1, with slippage:
```python
# BUY execution price — you pay slightly more than open (market impact)
exec_price = open_price * (1 + slippage_pct)   # e.g. 0.05% slippage

# SELL execution price — you receive slightly less than open
exec_price = open_price * (1 - slippage_pct)
```

### Sharpe Ratio from First Principles
```python
daily_rf       = annual_risk_free_rate / 252
excess_returns = daily_returns - daily_rf
sharpe         = (excess_returns.mean() / excess_returns.std()) * sqrt(252)
```

### Maximum Drawdown from First Principles
```python
running_peak = equity.cummax()
drawdown     = (equity - running_peak) / running_peak * 100
max_drawdown = drawdown.min()   # most negative value
```

---

## 🛠️ Dependencies

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | DataFrame operations, time series |
| `numpy` | ≥ 1.24 | Numerical computations |
| `yfinance` | ≥ 0.2 | NSE historical data (free) |
| `matplotlib` | ≥ 3.7 | Chart visualisation |

Install all at once:
```bash
pip install pandas numpy yfinance matplotlib
```

No third-party backtesting libraries (backtrader, zipline, pyfolio, quantstats) are used. All logic is implemented from scratch.

---

## 🔭 Roadmap / Future Work

-  Multi-asset portfolio support (trade basket of stocks simultaneously)
-  Walk-forward optimisation (prevent overfitting to historical data)
-  Order Book Simulator (Level 2 data, limit orders)
-  MACD-based strategy implementation
-  Intraday backtesting (minute-level OHLCV)
-  Parameter optimisation with grid search
-  Interactive Plotly dashboard

---

## 📚 Further Reading

- [Advances in Financial Machine Learning — Marcos López de Prado](https://www.amazon.in/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
- [Quantitative Trading — Ernest Chan](https://www.amazon.in/Quantitative-Trading-Build-Algorithmic-Business/dp/0470284889)
- [NSE India — Official Market Data](https://www.nseindia.com)
- [Brainstellar — Quant Interview Puzzles](https://brainstellar.com)

---

## 👤 Author

**Parth;)Khandelwal**
- 📧 parthkhandelwal1335@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/parth-khandelwal-24b33127b)


---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.