# Neural Quant

A comprehensive quantitative trading framework that combines traditional financial strategies with modern machine learning techniques for systematic trading and portfolio management.

## Overview

Neural Quant is a production-ready quantitative trading system designed for institutional and sophisticated retail traders. The framework provides a unified platform for strategy development, backtesting, risk management, and live trading across multiple asset classes and timeframes.

### Key Features

- **Multi-Strategy Framework**: Support for momentum, mean reversion, and machine learning-based strategies
- **Advanced ML Integration**: LSTM, Transformer, and ensemble models for predictive trading
- **Comprehensive Backtesting**: High-fidelity backtesting with realistic transaction costs and slippage
- **Risk Management**: Built-in position sizing, stop-loss, and portfolio-level risk controls
- **Multi-Asset Support**: Equities, ETFs, and crypto trading capabilities
- **Production Ready**: Paper trading and live trading with multiple broker integrations
- **Experiment Tracking**: MLflow integration for strategy performance monitoring
- **Modular Architecture**: Extensible design for custom strategies and data sources

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

4. **Configure the system**
   ```bash
   cp configs/config.example.yaml configs/config.yaml
   # Edit configs/config.yaml with your settings
   ```

5. **Verify installation**
   ```bash
   python scripts/verify_setup.py
   ```

### Basic Usage

#### 1. Run a Strategy Backtest

```bash
python scripts/run_strategy.py
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
                    │ • Position Sizing│
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
- **UI Dashboard**: Web interface for experiment analysis

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

### v1.0.0 (2024-12-01)
- Initial release
- Core strategy framework
- MLflow integration
- Basic backtesting engine
- Paper trading support

---

**Neural Quant** - Professional Quantitative Trading Framework

