"""
MLflow Manager for Neural Quant

This module provides a simplified MLflow manager for logging experiments,
parameters, metrics, and artifacts.
"""

import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MLflowManager:
    """Simplified MLflow manager for Neural Quant."""
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        """
        Initialize the MLflow manager.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        
    def start_run(self, run_name: Optional[str] = None, experiment_name: str = "neural_quant"):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            experiment_name: Name of the experiment
            
        Returns:
            MLflow run context manager
        """
        try:
            # Set experiment
            mlflow.set_experiment(experiment_name)
            
            # Start run
            return mlflow.start_run(run_name=run_name)
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            return self._dummy_context_manager()
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to MLflow.
        
        Args:
            local_path: Path to the local file
            artifact_path: Path within the artifact store
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")
    
    def _dummy_context_manager(self):
        """Return a dummy context manager for when MLflow fails."""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()

# Global instance
_mlflow_manager = None

def get_mlflow_manager() -> MLflowManager:
    """Get the global MLflow manager instance."""
    global _mlflow_manager
    if _mlflow_manager is None:
        _mlflow_manager = MLflowManager()
    return _mlflow_manager


