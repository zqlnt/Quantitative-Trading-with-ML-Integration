"""
Configuration management system for Neural Quant.

This module provides a centralized configuration management system that loads
configuration from YAML files and environment variables, with support for
different environments and secure credential handling.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class MLflowConfig(BaseModel):
    """MLflow configuration settings."""
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "neural_quant_experiments"
    artifact_root: str = "./mlruns"
    registry_uri: str = "sqlite:///mlflow.db"


class DataSourceConfig(BaseModel):
    """Data source configuration."""
    enabled: bool = False
    api_key: Optional[str] = None
    rate_limit: float = 1.0


class DataSourcesConfig(BaseModel):
    """All data sources configuration."""
    yfinance: DataSourceConfig = DataSourceConfig(enabled=True)
    alpha_vantage: DataSourceConfig = DataSourceConfig()
    polygon: DataSourceConfig = DataSourceConfig()
    iex_cloud: DataSourceConfig = DataSourceConfig()


class BrokerConfig(BaseModel):
    """Broker configuration."""
    enabled: bool = False
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: Optional[str] = None
    data_url: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    client_id: Optional[int] = None


class BrokersConfig(BaseModel):
    """All brokers configuration."""
    alpaca: BrokerConfig = BrokerConfig()
    interactive_brokers: BrokerConfig = BrokerConfig()


class TradingConfig(BaseModel):
    """Trading configuration."""
    default_symbols: list = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    default_timeframe: str = "1d"
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.15
    
    paper_trading: Dict[str, Any] = {
        "initial_capital": 100000,
        "commission": 0.0,
        "slippage": 0.001
    }


class RiskConfig(BaseModel):
    """Risk management configuration."""
    position_sizing: Dict[str, Any] = {
        "method": "fixed_percentage",
        "max_position_pct": 0.1
    }
    stop_loss: Dict[str, Any] = {
        "enabled": True,
        "default_pct": 0.05
    }
    take_profit: Dict[str, Any] = {
        "enabled": True,
        "default_pct": 0.15
    }
    circuit_breakers: Dict[str, Any] = {
        "max_daily_trades": 50,
        "max_concurrent_positions": 10,
        "max_daily_loss_pct": 0.05
    }


class ModelConfig(BaseModel):
    """Model configuration."""
    baseline: Dict[str, Any] = {
        "momentum": {"lookback_period": 20, "threshold": 0.02},
        "mean_reversion": {"lookback_period": 20, "threshold": 2.0},
        "logistic_regression": {
            "features": ["rsi", "macd", "bollinger_upper", "bollinger_lower"],
            "test_size": 0.2
        }
    }
    neural: Dict[str, Any] = {
        "lstm": {
            "sequence_length": 60,
            "hidden_units": 50,
            "dropout": 0.2,
            "epochs": 100,
            "batch_size": 32
        },
        "transformer": {
            "sequence_length": 60,
            "d_model": 64,
            "n_heads": 8,
            "n_layers": 4,
            "dropout": 0.1,
            "epochs": 100,
            "batch_size": 32
        }
    }


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    technical_indicators: Dict[str, Any] = {
        "rsi": {"period": 14},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "bollinger_bands": {"period": 20, "std_dev": 2},
        "sma": {"periods": [5, 10, 20, 50, 200]},
        "ema": {"periods": [5, 10, 20, 50, 200]}
    }
    fundamental_features: Dict[str, Any] = {
        "enabled": False,
        "include_ratios": True,
        "include_earnings": True
    }
    sentiment_features: Dict[str, Any] = {
        "enabled": False,
        "news_sources": ["newsapi", "reddit", "twitter"],
        "update_frequency": "1h"
    }


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: Dict[str, Any] = {
        "enabled": True,
        "log_dir": "./logs",
        "max_file_size": "10MB",
        "backup_count": 5
    }
    trade_logging: Dict[str, Any] = {
        "enabled": True,
        "log_trades": True,
        "log_signals": True,
        "log_performance": True
    }


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = "sqlite"
    path: str = "./data/neural_quant.db"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True


class NotificationsConfig(BaseModel):
    """Notifications configuration."""
    email: Dict[str, Any] = {"enabled": False}
    slack: Dict[str, Any] = {"enabled": False}
    discord: Dict[str, Any] = {"enabled": False}


class NeuralQuantConfig(BaseModel):
    """Main configuration class for Neural Quant."""
    environment: str = "development"
    mlflow: MLflowConfig = MLflowConfig()
    data_sources: DataSourcesConfig = DataSourcesConfig()
    brokers: BrokersConfig = BrokersConfig()
    trading: TradingConfig = TradingConfig()
    risk: RiskConfig = RiskConfig()
    models: ModelConfig = ModelConfig()
    features: FeaturesConfig = FeaturesConfig()
    logging: LoggingConfig = LoggingConfig()
    database: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig()
    notifications: NotificationsConfig = NotificationsConfig()

    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = ['development', 'paper_trading', 'live_trading']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v


class ConfigManager:
    """Configuration manager for loading and managing Neural Quant configuration."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, will look for
                        config.yaml in the configs directory.
        """
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[NeuralQuantConfig] = None
        self._load_environment_variables()
        
    def _find_config_file(self) -> Path:
        """Find the configuration file."""
        # Look for config.yaml first, then config.example.yaml
        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        
        config_file = config_dir / "config.yaml"
        if config_file.exists():
            return config_file
            
        example_file = config_dir / "config.example.yaml"
        if example_file.exists():
            print(f"Warning: Using example config file. Please copy {example_file} to config.yaml and update with your settings.")
            return example_file
            
        raise FileNotFoundError(f"No configuration file found in {config_dir}")
    
    def _load_environment_variables(self):
        """Load environment variables from .env file if it exists."""
        env_file = Path(__file__).parent.parent.parent.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    
    def load_config(self) -> NeuralQuantConfig:
        """
        Load configuration from file and environment variables.
        
        Returns:
            NeuralQuantConfig: Loaded configuration object.
        """
        if self._config is not None:
            return self._config
            
        # Load YAML configuration
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Override with environment variables
        config_data = self._override_with_env_vars(config_data)
        
        # Create configuration object
        self._config = NeuralQuantConfig(**config_data)
        return self._config
    
    def _override_with_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override configuration with environment variables.
        
        Args:
            config_data: Configuration data from YAML file.
            
        Returns:
            Dict[str, Any]: Configuration data with environment variable overrides.
        """
        # MLflow overrides
        if os.getenv('MLFLOW_TRACKING_URI'):
            config_data.setdefault('mlflow', {})['tracking_uri'] = os.getenv('MLFLOW_TRACKING_URI')
        if os.getenv('MLFLOW_EXPERIMENT_NAME'):
            config_data.setdefault('mlflow', {})['experiment_name'] = os.getenv('MLFLOW_EXPERIMENT_NAME')
            
        # API key overrides
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            config_data.setdefault('data_sources', {}).setdefault('alpha_vantage', {})['api_key'] = os.getenv('ALPHA_VANTAGE_API_KEY')
        if os.getenv('POLYGON_API_KEY'):
            config_data.setdefault('data_sources', {}).setdefault('polygon', {})['api_key'] = os.getenv('POLYGON_API_KEY')
        if os.getenv('IEX_CLOUD_API_KEY'):
            config_data.setdefault('data_sources', {}).setdefault('iex_cloud', {})['api_key'] = os.getenv('IEX_CLOUD_API_KEY')
            
        # Broker overrides
        if os.getenv('ALPACA_API_KEY'):
            config_data.setdefault('brokers', {}).setdefault('alpaca', {})['api_key'] = os.getenv('ALPACA_API_KEY')
        if os.getenv('ALPACA_SECRET_KEY'):
            config_data.setdefault('brokers', {}).setdefault('alpaca', {})['secret_key'] = os.getenv('ALPACA_SECRET_KEY')
            
        # Environment override
        if os.getenv('NEURAL_QUANT_ENV'):
            config_data['environment'] = os.getenv('NEURAL_QUANT_ENV')
            
        return config_data
    
    def get_config(self) -> NeuralQuantConfig:
        """
        Get the current configuration.
        
        Returns:
            NeuralQuantConfig: Current configuration object.
        """
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> NeuralQuantConfig:
        """
        Reload configuration from file.
        
        Returns:
            NeuralQuantConfig: Reloaded configuration object.
        """
        self._config = None
        return self.load_config()
    
    def get_data_source_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """
        Get configuration for a specific data source.
        
        Args:
            source_name: Name of the data source.
            
        Returns:
            Optional[DataSourceConfig]: Configuration for the data source, or None if not found.
        """
        config = self.get_config()
        return getattr(config.data_sources, source_name, None)
    
    def get_broker_config(self, broker_name: str) -> Optional[BrokerConfig]:
        """
        Get configuration for a specific broker.
        
        Args:
            broker_name: Name of the broker.
            
        Returns:
            Optional[BrokerConfig]: Configuration for the broker, or None if not found.
        """
        config = self.get_config()
        return getattr(config.brokers, broker_name, None)
    
    def is_environment(self, env: str) -> bool:
        """
        Check if the current environment matches the specified environment.
        
        Args:
            env: Environment to check.
            
        Returns:
            bool: True if the current environment matches.
        """
        return self.get_config().environment == env
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.is_environment('development')
    
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading environment."""
        return self.is_environment('paper_trading')
    
    def is_live_trading(self) -> bool:
        """Check if running in live trading environment."""
        return self.is_environment('live_trading')


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> NeuralQuantConfig:
    """
    Get the global configuration instance.
    
    Returns:
        NeuralQuantConfig: Global configuration object.
    """
    return config_manager.get_config()


def get_data_source_config(source_name: str) -> Optional[DataSourceConfig]:
    """
    Get configuration for a specific data source.
    
    Args:
        source_name: Name of the data source.
        
    Returns:
        Optional[DataSourceConfig]: Configuration for the data source, or None if not found.
    """
    return config_manager.get_data_source_config(source_name)


def get_broker_config(broker_name: str) -> Optional[BrokerConfig]:
    """
    Get configuration for a specific broker.
    
    Args:
        broker_name: Name of the broker.
        
    Returns:
        Optional[BrokerConfig]: Configuration for the broker, or None if not found.
    """
    return config_manager.get_broker_config(broker_name)
