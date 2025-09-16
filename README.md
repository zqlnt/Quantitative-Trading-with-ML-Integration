# Neural Quant

A professional quantitative trading framework for systematic strategy development, backtesting, and experiment tracking.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/zqlnt/Quantitative-Trading-with-ML-Integration.git
cd Quantitative-Trading-with-ML-Integration
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start the dashboard
streamlit run src/neural_quant/ui/app.py --server.port 8502
```

Open http://localhost:8502 in your browser.

## ✨ Features

- **4 Trading Strategies**: Moving Average Crossover, Bollinger Bands, Volatility Breakout, Cross-Sectional Momentum
- **High-Fidelity Backtesting**: Realistic transaction costs, slippage, and portfolio management
- **Statistical Analysis**: Monte Carlo Permutation Tests, Bootstrap Confidence Intervals, Walk-Forward Analysis
- **Interactive Dashboard**: Streamlit UI for strategy configuration and analysis
- **Market Watch**: Live price charts for 12 supported tickers
- **AI Assistant**: Anthropic Claude integration for strategy analysis
- **Experiment Tracking**: MLflow integration for reproducible research
- **Strategy Iteration**: R1→R2→Execution→R3 loop for automated strategy improvement
- **Risk Management**: Regime filtering, volatility targeting, position caps, and basic exits
- **Portfolio Management**: Equal-weight and volatility-weighted allocation methods

## 📊 Usage

### Streamlit Dashboard
1. **Backtest Results**: Run strategies with statistical significance testing and Strategy Analyst evaluation
2. **Walk-Forward Analysis**: Rolling window analysis with significance tracking
3. **Strategy Library**: Browse available strategies and parameters
4. **Metrics Key**: Learn about performance metrics and statistical concepts
5. **Market Watch**: Monitor live prices and market data
6. **Strategy Iteration**: Automated R1→R2→Execution→R3 loop for failed strategies

### Command Line
```bash
# Single-ticker backtest
python -m neural_quant.scripts.run_strategy --strategy momentum --symbols AAPL --timeframe 1d --years 3

# Portfolio backtest
python -m neural_quant.scripts.run_strategy --strategy cross_sectional_momentum --symbols AAPL,TSLA,NVDA --timeframe 1d --years 2
```

### Programmatic Usage
```python
from neural_quant.core.backtest import Backtester
from neural_quant.strategies.momentum import MovingAverageCrossover
from neural_quant.data.yf_loader import load_yf_data

# Load data and create strategy
data = load_yf_data(['AAPL'], start_date='2023-01-01', end_date='2024-01-01')
strategy = MovingAverageCrossover(ma_fast=10, ma_slow=30, threshold=0.0)

# Run backtest
backtester = Backtester(commission=0.001, slippage=0.0002)
results = backtester.run_backtest(data, strategy, '2023-01-01', '2024-01-01')

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## 🏗️ Architecture

```
Data Layer → Strategy Layer → Execution Layer
     ↓              ↓              ↓
  Analytics Layer (MLflow, Streamlit, AI Assistant)
```

## 📁 Project Structure

```
neural-quant/
├── src/neural_quant/
│   ├── core/                   # Backtesting engine
│   ├── strategies/             # Trading strategies
│   ├── data/                   # Data loaders
│   ├── ui/                     # Streamlit dashboard
│   ├── analysis/               # Statistical analysis
│   └── scripts/                # Command-line tools
├── requirements.txt
└── pyproject.toml
```

## ⚙️ Configuration

### Environment Variables
```bash
# Optional: Anthropic API key for AI assistant
export ANTHROPIC_API_KEY=your_api_key_here
```

### Strategy Parameters
- **Moving Average Crossover**: `ma_fast`, `ma_slow`, `threshold`, `min_volume`, `max_positions`
- **Bollinger Bands**: `window`, `num_std`
- **Volatility Breakout**: `atr_window`, `multiplier`
- **Cross-Sectional Momentum**: `lookback_window`, `top_n`, `bottom_n`

## 📈 Performance Metrics

- **Returns**: Total return, CAGR, annualized return
- **Risk**: Volatility, maximum drawdown, VaR, CVaR
- **Ratios**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trading**: Win rate, profit factor, average trade metrics
- **Portfolio**: Turnover, position counts, per-symbol performance

## 🔧 Development

### Adding New Strategies
```python
from neural_quant.strategies.base.strategy_base import StrategyBase

class MyStrategy(StrategyBase):
    def __init__(self, **kwargs):
        super().__init__(name="MyStrategy", **kwargs)
    
    def _generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        # Your strategy logic here
        pass
```

### Running Tests
```bash
pytest                    # Run all tests
pytest --cov=src         # Run with coverage
```

## 🐛 Troubleshooting

**Data Loading Errors**
- Some symbols may not be available (e.g., XAUUSD=X). The system automatically falls back to alternatives (GC=F).

**Import Errors**
- Always use `python -m neural_quant.scripts.script_name` instead of direct script execution.
- Ensure the package is installed: `pip install -e .`

**MLflow Issues**
- Start MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`

**Streamlit Issues**
- Ensure port 8502 is available for the dashboard.

## 🤖 AI-Powered Strategy Development

### R1 - Quant Researcher
- **Role**: Analyzes failed runs and proposes improvement experiments
- **Input**: Run artifacts (metrics, p-values, confidence intervals)
- **Output**: Up to 3 testable experiments with success criteria
- **Focus**: Robustness improvements over just returns

### R2 - Strategy Developer
- **Role**: Translates experiments into precise change requests
- **Input**: Selected experiments from R1
- **Output**: Parameter grids, overlay settings, code changes
- **Focus**: Minimal changes with maximum impact

### R3 - Strategy Analyst
- **Role**: Final arbiter for run evaluation and promotion decisions
- **Input**: Full Run Bundle artifacts
- **Output**: Executive summary, significance verdict, promotion decision
- **Focus**: Strict adherence to promotion rules

### Orchestration Engine
- **Role**: Manages complete R1→R2→Execution→R3 loop
- **Features**: Automated experiment execution, lineage tracking, iteration graphs
- **UI**: "Iterate from this run" button for rejected strategies

## 📋 Changelog

### v0.2.0 (Current)
- **AI-Powered Strategy Development**: R1→R2→R3 loop with automated iteration
- **Strategy Analyst**: Final arbiter for promotion decisions
- **Risk Management**: Regime filtering, volatility targeting, position caps
- **Portfolio Management**: Equal-weight and volatility-weighted allocation
- **Basic Exits**: ATR stops and time-based stops
- **Artifact Management**: Structured Run Bundle with all artifacts
- **Iteration Tracking**: Parent-child run relationships and lineage graphs

### v0.1.0
- 4 trading strategies with high-fidelity backtesting
- Monte Carlo Permutation Testing and Bootstrap Confidence Intervals
- Walk-Forward Analysis with significance tracking
- Streamlit dashboard with interactive UI
- Market Watch with live price data
- AI Assistant integration
- MLflow experiment tracking

## 🗺️ Roadmap

- Additional trading strategies (ML-based, mean reversion)
- Extended asset class support
- Live trading integration
- Real-time data streaming
- Advanced portfolio optimization

## 📄 License

MIT License

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

---

**Neural Quant** - Professional Quantitative Trading Framework