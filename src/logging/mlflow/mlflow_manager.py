"""
MLflow integration for Neural Quant.

This module provides comprehensive MLflow integration for experiment tracking,
model versioning, and performance monitoring across all trading strategies and models.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from ...utils.config.config_manager import get_config


class MLflowManager:
    """
    MLflow manager for Neural Quant experiment tracking.
    
    This class provides a centralized interface for MLflow operations including
    experiment management, run tracking, model logging, and artifact storage.
    """
    
    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize the MLflow manager.
        
        Args:
            experiment_name: Name of the experiment. If None, uses config default.
        """
        self.config = get_config()
        self.experiment_name = experiment_name or self.config.mlflow.experiment_name
        self.tracking_uri = self.config.mlflow.tracking_uri
        self.artifact_root = self.config.mlflow.artifact_root
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if it doesn't exist
        self._setup_experiment()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def _setup_experiment(self):
        """Set up the MLflow experiment."""
        try:
            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=self.artifact_root
                )
                self.logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                self.logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
                
            self.experiment_id = experiment_id
            
        except Exception as e:
            self.logger.error(f"Failed to set up MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run. If None, auto-generated.
            tags: Dictionary of tags to add to the run.
            
        Returns:
            mlflow.ActiveRun: Active MLflow run object.
        """
        tags = tags or {}
        tags.update({
            "project": "neural_quant",
            "environment": self.config.environment,
            "timestamp": datetime.now().isoformat()
        })
        
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log.
        """
        try:
            mlflow.log_params(params)
            self.logger.debug(f"Logged parameters: {list(params.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number for the metrics.
        """
        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
            self.logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_artifacts(self, local_dir: Union[str, Path], artifact_path: Optional[str] = None):
        """
        Log artifacts from a local directory.
        
        Args:
            local_dir: Local directory containing artifacts.
            artifact_path: Optional path within the artifact store.
        """
        try:
            mlflow.log_artifacts(str(local_dir), artifact_path)
            self.logger.debug(f"Logged artifacts from: {local_dir}")
        except Exception as e:
            self.logger.error(f"Failed to log artifacts: {e}")
            raise
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None):
        """
        Log a single artifact file.
        
        Args:
            local_path: Path to the local file.
            artifact_path: Optional path within the artifact store.
        """
        try:
            mlflow.log_artifact(str(local_path), artifact_path)
            self.logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            self.logger.error(f"Failed to log artifact: {e}")
            raise
    
    def log_model(self, model: Any, model_name: str, model_type: str = "sklearn"):
        """
        Log a trained model to MLflow.
        
        Args:
            model: The trained model object.
            model_name: Name for the model.
            model_type: Type of model (sklearn, pytorch, xgboost).
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, model_name)
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, model_name)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            self.logger.info(f"Logged {model_type} model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to log model: {e}")
            raise
    
    def log_dataframe(self, df: pd.DataFrame, name: str, format: str = "csv"):
        """
        Log a pandas DataFrame as an artifact.
        
        Args:
            df: DataFrame to log.
            name: Name for the artifact.
            format: Format to save the DataFrame (csv, parquet).
        """
        try:
            if format == "csv":
                df.to_csv(f"{name}.csv", index=False)
                mlflow.log_artifact(f"{name}.csv")
                os.remove(f"{name}.csv")  # Clean up temporary file
            elif format == "parquet":
                df.to_parquet(f"{name}.parquet", index=False)
                mlflow.log_artifact(f"{name}.parquet")
                os.remove(f"{name}.parquet")  # Clean up temporary file
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.debug(f"Logged DataFrame as {format}: {name}")
        except Exception as e:
            self.logger.error(f"Failed to log DataFrame: {e}")
            raise
    
    def log_trading_results(self, results: Dict[str, Any]):
        """
        Log trading strategy results.
        
        Args:
            results: Dictionary containing trading results and metrics.
        """
        try:
            # Log performance metrics
            if "performance_metrics" in results:
                self.log_metrics(results["performance_metrics"])
            
            # Log strategy parameters
            if "strategy_params" in results:
                self.log_parameters(results["strategy_params"])
            
            # Log trade data if available
            if "trades" in results and isinstance(results["trades"], pd.DataFrame):
                self.log_dataframe(results["trades"], "trades")
            
            # Log portfolio data if available
            if "portfolio" in results and isinstance(results["portfolio"], pd.DataFrame):
                self.log_dataframe(results["portfolio"], "portfolio")
            
            # Log additional artifacts
            if "artifacts" in results:
                for artifact_name, artifact_data in results["artifacts"].items():
                    if isinstance(artifact_data, pd.DataFrame):
                        self.log_dataframe(artifact_data, artifact_name)
                    else:
                        self.log_artifact(artifact_data, artifact_name)
            
            self.logger.info("Logged trading results successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to log trading results: {e}")
            raise
    
    def log_model_performance(self, model_name: str, performance: Dict[str, float], 
                            model_params: Dict[str, Any], feature_importance: Optional[pd.DataFrame] = None):
        """
        Log model performance metrics and parameters.
        
        Args:
            model_name: Name of the model.
            performance: Dictionary of performance metrics.
            model_params: Dictionary of model parameters.
            feature_importance: Optional DataFrame with feature importance.
        """
        try:
            # Log model parameters
            self.log_parameters({f"{model_name}_{k}": v for k, v in model_params.items()})
            
            # Log performance metrics
            self.log_metrics({f"{model_name}_{k}": v for k, v in performance.items()})
            
            # Log feature importance if available
            if feature_importance is not None:
                self.log_dataframe(feature_importance, f"{model_name}_feature_importance")
            
            self.logger.info(f"Logged performance for model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to log model performance: {e}")
            raise
    
    def get_experiment_runs(self, filter_string: Optional[str] = None) -> List[mlflow.entities.Run]:
        """
        Get all runs from the current experiment.
        
        Args:
            filter_string: Optional MLflow filter string.
            
        Returns:
            List[mlflow.entities.Run]: List of runs.
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string
            )
            return runs
        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {e}")
            raise
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare.
            
        Returns:
            pd.DataFrame: Comparison DataFrame.
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"run_id in ({','.join(run_ids)})"
            )
            return runs
        except Exception as e:
            self.logger.error(f"Failed to compare runs: {e}")
            raise
    
    def get_best_run(self, metric_name: str, ascending: bool = False) -> Optional[mlflow.entities.Run]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize.
            ascending: Whether to sort in ascending order.
            
        Returns:
            Optional[mlflow.entities.Run]: Best run or None if no runs found.
        """
        try:
            runs = self.get_experiment_runs()
            if runs.empty:
                return None
            
            if metric_name not in runs.columns:
                self.logger.warning(f"Metric {metric_name} not found in runs")
                return None
            
            # Sort by metric
            sorted_runs = runs.sort_values(metric_name, ascending=ascending)
            best_run_id = sorted_runs.iloc[0]['run_id']
            
            # Get the actual run object
            return mlflow.get_run(best_run_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get best run: {e}")
            return None
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current run.
        
        Args:
            status: Status of the run (FINISHED, FAILED, KILLED).
        """
        try:
            mlflow.end_run(status=status)
            self.logger.info(f"Ended MLflow run with status: {status}")
        except Exception as e:
            self.logger.error(f"Failed to end run: {e}")
            raise


# Global MLflow manager instance
mlflow_manager = MLflowManager()


def get_mlflow_manager() -> MLflowManager:
    """
    Get the global MLflow manager instance.
    
    Returns:
        MLflowManager: Global MLflow manager instance.
    """
    return mlflow_manager


def start_experiment_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
    """
    Start a new experiment run using the global MLflow manager.
    
    Args:
        run_name: Name for the run.
        tags: Tags for the run.
        
    Returns:
        mlflow.ActiveRun: Active MLflow run.
    """
    return mlflow_manager.start_run(run_name, tags)


def log_trading_strategy_results(strategy_name: str, results: Dict[str, Any], 
                               strategy_params: Dict[str, Any]):
    """
    Log results from a trading strategy.
    
    Args:
        strategy_name: Name of the strategy.
        results: Dictionary containing strategy results.
        strategy_params: Dictionary containing strategy parameters.
    """
    with mlflow_manager.start_run(run_name=f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log strategy parameters
        mlflow_manager.log_parameters(strategy_params)
        
        # Log trading results
        mlflow_manager.log_trading_results(results)
        
        # Add strategy-specific tags
        mlflow.set_tag("strategy_name", strategy_name)
        mlflow.set_tag("run_type", "strategy_evaluation")


def log_model_training(model_name: str, model: Any, model_type: str, 
                      performance: Dict[str, float], model_params: Dict[str, Any],
                      feature_importance: Optional[pd.DataFrame] = None):
    """
    Log model training results.
    
    Args:
        model_name: Name of the model.
        model: Trained model object.
        model_type: Type of model (sklearn, pytorch, xgboost).
        performance: Dictionary of performance metrics.
        model_params: Dictionary of model parameters.
        feature_importance: Optional DataFrame with feature importance.
    """
    with mlflow_manager.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log model
        mlflow_manager.log_model(model, model_name, model_type)
        
        # Log performance
        mlflow_manager.log_model_performance(model_name, performance, model_params, feature_importance)
        
        # Add model-specific tags
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("run_type", "model_training")
