# Neural Quant

A professional quantitative trading framework for systematic strategy development, backtesting, and experiment tracking. Built with Python, featuring MLflow integration and a Streamlit dashboard for interactive analysis.

## Current Status

**FULLY OPERATIONAL** - All core systems working and tested

### What's Working
- **Complete Package Structure**: Modern `src/` layout with proper imports
- **Data Loading**: Yahoo Finance integration with timezone normalization
- **Trading Strategies**: Moving Average Crossover strategy with robust signal generation
- **Backtesting Engine**: High-fidelity backtesting with realistic transaction costs
- **MLflow Integration**: Automatic experiment tracking and artifact storage
- **Streamlit Dashboard**: Interactive UI for strategy configuration and results analysis
- **Timezone Handling**: Comprehensive timezone normalization for daily/intraday data
- **Testing Suite**: Unit tests for all major components

### What's Implemented
- **Core Framework**: Backtesting engine, metrics calculation, data loaders
- **Strategy System**: Momentum-based strategies with configurable parameters
- **Experiment Tracking**: MLflow integration with parameter/metric logging
- **Web Interface**: Streamlit dashboard for interactive strategy testing
- **Data Pipeline**: Yahoo Finance data loading with proper timezone handling
- **Configuration System**: YAML-based configuration management

### What's Ready for Development
- **Additional Strategies**: Mean reversion, ML-based strategies
- **Live Trading**: Broker integration for paper/live trading
- **Advanced Analytics**: Additional performance metrics and risk measures
- **Multi-Asset Support**: Extended asset class coverage

## Package Structure

Neural Quant uses a modern Python package structure with `src/` layout:

```
neural-quant/
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
├── src/
│   └── neural_quant/           # Main package
│       ├── core/               # Backtesting engine, metrics
│       │   ├── backtest.py     # High-fidelity backtesting
│       │   └── metrics.py      # Performance metrics
│       ├── data/               # Data loaders
│       │   └── yf_loader.py    # Yahoo Finance integration
│       ├── strategies/         # Trading strategies
│       │   └── momentum.py     # MA Crossover strategy
│       ├── scripts/            # Command-line tools
│       │   ├── run_strategy.py # Strategy execution
│       │   └── verify_setup.py # Setup verification
│       ├── ui/                 # User interface
│       │   └── app.py          # Streamlit dashboard
│       └── utils/              # Utilities
│           └── time_utils.py   # Timezone handling
├── configs/                    # Configuration files
├── tests/                      # Test suite
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/neural-quant.git
   cd neural-quant
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode**
   ```bash
   pip install -e .
   ```

5. **Verify installation**
   ```bash
   python -m neural_quant.scripts.verify_setup
   ```

### Using the Streamlit Dashboard

The easiest way to get started is with the interactive Streamlit dashboard:

1. **Start the dashboard**
   ```bash
   streamlit run src/neural_quant/ui/app.py --server.port 8502
   ```

2. **Open in browser**
   - Navigate to http://localhost:8502
   - Configure your strategy parameters in the sidebar
   - Click "Run Backtest" to execute and analyze results

### Using MLflow for Experiment Tracking

1. **Start MLflow UI**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   ```

2. **View experiments**
   - Navigate to http://localhost:5000
   - Browse all backtest runs and results
   - Compare strategy performance across different parameters

6. **Run your first strategy**
   ```bash
   source venv/bin/activate
   python -m neural_quant.scripts.run_strategy --strategy momentum --symbols AAPL --timeframe 1d --years 1
   ```

7. **Verify installation**
   ```bash
   source venv/bin/activate
   python -m neural_quant.scripts.verify_setup
   ```

### Basic Usage

**Important**: Always use module execution (`python -m`) instead of direct script execution to avoid import issues.

#### 1. Run a Strategy Backtest

```bash
# Basic momentum strategy
python -m neural_quant.scripts.run_strategy --strategy momentum --symbols AAPL --timeframe 1d --years 1

# Advanced parameters
python -m neural_quant.scripts.run_strategy --strategy momentum --symbols AAPL MSFT GOOGL --timeframe 1d --years 3 --ma_fast 10 --ma_slow 30 --threshold 0.0
```

#### 2. Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Visit http://localhost:5000 to view experiments
```

#### 3. Run Sanity Checks

```bash
python scripts/run_sanity_checks.py

# Run a strategy backtest
python scripts/run_strategy.py
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Strategy Layer │    │ Execution Layer │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • YFinance      │    │ • Momentum      │    │ • Paper Trading │
│ • Alpha Vantage │    │ • Mean Reversion│    │ • Alpaca        │
│ • Polygon       │    │ • ML Models     │    │ • Interactive   │
│ • Custom Feeds  │    │ • Portfolio     │    │   Brokers       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Risk & Control │
                    ├─────────────────┤
                    │ • Position Size │
                    │ • Stop Loss     │
                    │ • Circuit Breaks│
                    │ • Portfolio Risk│
                    └─────────────────┘
   ```

## Project Structure

```
neural-quant/
├── src/                    # Source code
│   ├── data/              # Data collection and processing
│   ├── strategies/        # Trading strategies
│   ├── models/            # ML models
│   ├── execution/         # Order execution
│   ├── risk/              # Risk management
│   ├── logging/           # Logging and tracking
│   └── utils/             # Utilities and helpers
├── configs/               # Configuration files
├── scripts/               # Executable scripts
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
├── docs/                  # Documentation
└── experiments/           # Experiment results
```

## Configuration

The system uses a hierarchical configuration system with environment-specific settings:

### Environment Variables

```bash
# Optional: Set environment
export NEURAL_QUANT_ENV=development

# Optional: Set custom config path
export NEURAL_QUANT_CONFIG=path/to/your/config.yaml
```

### Configuration File Structure

```yaml
# configs/config.yaml
environment: development  # development, paper_trading, live_trading

# Data Sources
data_sources:
  yfinance:
    enabled: true
    rate_limit: 1.0
  alpha_vantage:
    enabled: false
    api_key: "YOUR_API_KEY"
  polygon:
    enabled: false
    api_key: "YOUR_API_KEY"

# Trading Configuration
trading:
  default_symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
  default_timeframe: "1d"
  max_position_size: 0.1
  max_daily_loss: 0.05

# Risk Management
risk:
  position_sizing:
    method: "fixed_percentage"
    max_position_pct: 0.1
  stop_loss:
    enabled: true
    default_pct: 0.05

# MLflow Tracking
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "neural_quant_experiments"
  artifact_root: "./mlruns"
```

## Strategy Development

### Creating a Custom Strategy

```python
from src.strategies.base.strategy_base import StrategyBase
from src.utils.validation.metrics import calculate_metrics

class MyCustomStrategy(StrategyBase):
    def __init__(self, config: dict):
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 20)
        self.threshold = config.get('threshold', 0.02)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on custom logic."""
        # Your strategy logic here
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['strength'] = 0.0
        
        # Example: Simple moving average crossover
        sma_short = data['close'].rolling(self.lookback_period//2).mean()
        sma_long = data['close'].rolling(self.lookback_period).mean()
        
        signals.loc[sma_short > sma_long, 'signal'] = 1
        signals.loc[sma_short < sma_long, 'signal'] = -1
        signals['strength'] = abs(sma_short - sma_long) / data['close']
        
        return signals
```

### Available Strategies

- **MomentumStrategy**: Price momentum and trend following
- **MeanReversionStrategy**: Mean reversion and contrarian signals
- **LogisticRegressionStrategy**: ML-based classification
- **LSTMStrategy**: Deep learning time series prediction
- **TransformerStrategy**: Attention-based sequence modeling

## Data Sources

### Yahoo Finance (Default)
- **Cost**: Free
- **Coverage**: Global equities, ETFs, indices
- **Rate Limit**: 1 request/second (configurable)
- **Setup**: No API key required

### Alpha Vantage
- **Cost**: Free tier available
- **Coverage**: US equities, forex, crypto
- **Rate Limit**: 5 requests/minute (free tier)
- **Setup**: Requires API key

### Polygon.io
- **Cost**: Free tier available
- **Coverage**: US equities, options, forex
- **Rate Limit**: 5 requests/minute (free tier)
- **Setup**: Requires API key

## Risk Management

### Position Sizing
- **Fixed Percentage**: Allocate fixed % of portfolio per position
- **Volatility-Based**: Size based on asset volatility
- **Kelly Criterion**: Optimal position sizing based on win rate and payoff

### Risk Controls
- **Stop Loss**: Automatic position exit on adverse moves
- **Take Profit**: Systematic profit taking
- **Circuit Breakers**: Halt trading on excessive losses
- **Portfolio Limits**: Maximum exposure and correlation controls

## Backtesting

### Running Backtests

```python
from src.utils.validation.backtesting import BacktestEngine
from src.strategies.baselines.momentum_strategy import MomentumStrategy

# Initialize strategy
strategy = MomentumStrategy({
    'lookback_period': 20,
    'threshold': 0.02
})

# Run backtest
engine = BacktestEngine(strategy)
results = engine.run(
    symbols=['AAPL', 'MSFT'],
    start_date='2023-01-01',
    end_date='2024-01-01',
    initial_capital=100000
)

# Analyze results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Performance Metrics

- **Returns**: Total, annualized, and risk-adjusted returns
- **Risk**: Volatility, VaR, CVaR, and maximum drawdown
- **Ratios**: Sharpe, Sortino, Calmar, and information ratios
- **Trading**: Win rate, profit factor, and average trade metrics

## Live Trading

### Paper Trading

```bash
# Start paper trading
python scripts/paper_trading.py --strategy momentum --symbols AAPL,MSFT
```

### Live Trading (Alpaca)

1. **Get API credentials** from [Alpaca Markets](https://alpaca.markets/)
2. **Update configuration**:
   ```yaml
   brokers:
     alpaca:
       enabled: true
       api_key: "YOUR_API_KEY"
       secret_key: "YOUR_SECRET_KEY"
       base_url: "https://paper-api.alpaca.markets"  # Paper trading
   ```
3. **Start live trading**:
   ```bash
   python scripts/live_trading.py --strategy momentum
   ```

## Monitoring and Logging

### MLflow Integration

- **Experiment Tracking**: Automatic logging of strategy parameters and results
- **Model Registry**: Version control for trained models
- **Artifact Storage**: Save and retrieve strategy artifacts
- **UI Dashboard**: Web interface for experiment analysis at `http://localhost:5000`

### Streamlit Dashboard

A comprehensive web-based dashboard for experiment analysis and strategy monitoring:

**Features:**
- **Real-time MLflow Integration**: Direct SQLite database connection
- **Advanced Filtering**: Filter by symbol, strategy, gate, Sharpe ratio, drawdown
- **Performance Leaderboard**: Ranked by Sharpe_post_cost, MaxDD, Turnover, VaR/CVaR
- **Run Detail Analysis**: Parameters, metrics, metadata, and artifact previews
- **Visual Artifacts**: Equity curves, price+MA charts, CSV data tables
- **Interactive Charts**: Built-in Streamlit plotting capabilities
- **File Browser**: Complete artifact directory listing

**Access:**
```bash
streamlit run app.py
# Dashboard available at: http://localhost:8501
```

**Environment Variables:**
```bash
# Custom MLflow database path
MLFLOW_DB_PATH=/path/to/mlflow.db streamlit run app.py

# Custom artifact root
MLFLOW_ARTIFACT_ROOT=/path/to/mlruns streamlit run app.py
```

### Logging

- **Trade Logs**: Detailed transaction records in `logs/trades.log`
- **Performance Logs**: Strategy performance metrics
- **Error Logs**: System errors and warnings
- **Database**: SQLite database for structured data storage

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_strategies.py
```

### Code Quality

```bash
# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/

# Type checking
mypy src/
```

## Performance Considerations

### Optimization Tips

1. **Data Caching**: Enable data caching to reduce API calls
2. **Parallel Processing**: Use multiprocessing for backtesting
3. **Memory Management**: Monitor memory usage for large datasets
4. **Database Indexing**: Optimize database queries for large datasets

### System Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4+ CPU cores
- **Storage**: 10GB+ for data and experiments
- **Network**: Stable internet for data feeds

## Project Summary

### What's Working (Fully Operational)

**Core Framework:**
- **Package Structure**: Modern `src/` layout with proper Python imports
- **Data Loading**: Yahoo Finance integration with timezone normalization
- **Backtesting Engine**: High-fidelity backtesting with realistic transaction costs
- **Strategy System**: Moving Average Crossover strategy with robust signal generation
- **MLflow Integration**: Automatic experiment tracking and artifact storage
- **Streamlit Dashboard**: Interactive UI for strategy configuration and analysis
- **Timezone Handling**: Comprehensive timezone normalization for daily/intraday data
- **Testing Suite**: Unit tests for all major components

**User Interface:**
- **Streamlit Dashboard**: Clean, professional interface at http://localhost:8502
- **Sidebar Configuration**: Market data, strategy parameters, transaction costs
- **Results Display**: Metrics, charts, trade logs, MLflow integration
- **MLflow UI**: Experiment tracking at http://localhost:5000

**Data & Strategies:**
- **Yahoo Finance Data**: Real-time and historical market data
- **Moving Average Strategy**: Configurable fast/slow MA periods and thresholds
- **Transaction Costs**: Realistic commission and slippage modeling
- **Performance Metrics**: Sharpe ratio, max drawdown, total return, etc.

### Technical Implementation

**Architecture:**
- Modern Python package structure with `pyproject.toml`
- Absolute imports with `python -m` module execution
- Comprehensive error handling and logging
- Timezone-aware data processing
- MLflow experiment tracking with SQLite backend

**Key Files:**
- `src/neural_quant/core/backtest.py` - High-fidelity backtesting engine
- `src/neural_quant/strategies/momentum.py` - Moving Average Crossover strategy
- `src/neural_quant/data/yf_loader.py` - Yahoo Finance data loader
- `src/neural_quant/ui/app.py` - Streamlit dashboard
- `src/neural_quant/utils/time_utils.py` - Timezone handling utilities

### Ready for Development

**Immediate Extensions:**
- Additional trading strategies (mean reversion, ML-based)
- Extended asset class support (crypto, forex, commodities)
- Advanced risk management features
- Live trading integration with brokers

**No API Keys Required:**
- Uses free Yahoo Finance data
- Local MLflow tracking (SQLite)
- No external service dependencies

### Getting Started

1. **Quick Start**: Use the Streamlit dashboard for interactive strategy testing
2. **Command Line**: Use `python -m neural_quant.scripts.run_strategy` for batch processing
3. **Experiment Tracking**: Use MLflow UI to analyze and compare results
4. **Development**: Extend with custom strategies and data sources

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Add type hints
- Use meaningful commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making investment decisions.

## Support

- **Documentation**: [Wiki](https://github.com/your-org/neural-quant/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-org/neural-quant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/neural-quant/discussions)
- **Email**: support@neural-quant.com

## Changelog

### v0.1.0 (Current)
- Complete package structure with `src/` layout
- High-fidelity backtesting engine with transaction costs
- Moving Average Crossover strategy implementation
- Yahoo Finance data integration with timezone handling
- MLflow experiment tracking and artifact storage
- Streamlit dashboard for interactive analysis
- Comprehensive testing suite
- Professional documentation and README

---

**Neural Quant** - Professional Quantitative Trading Framework

