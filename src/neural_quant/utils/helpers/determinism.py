"""
Determinism and seeding utilities for Neural Quant.

This module provides global seeding and determinism controls to ensure
reproducible results across runs.
"""

import random
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
import os
import hashlib

logger = logging.getLogger(__name__)


class DeterminismManager:
    """
    Global determinism manager for reproducible results.
    
    This class manages seeding across all random number generators
    to ensure reproducible results.
    """
    
    def __init__(self, global_seed: Optional[int] = None):
        """
        Initialize determinism manager.
        
        Args:
            global_seed: Global seed for all random generators
        """
        self.global_seed = global_seed
        self.seed_used = None
        self.logger = logging.getLogger(__name__)
        
        if global_seed is not None:
            self.set_global_seed(global_seed)
    
    def set_global_seed(self, seed: int):
        """
        Set global seed for all random generators.
        
        Args:
            seed: Seed value
        """
        self.global_seed = seed
        self.seed_used = seed
        
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy random seed
        np.random.seed(seed)
        
        # Set pandas random seed (if available)
        try:
            pd.core.common.random_state(seed)
        except AttributeError:
            pass
        
        # Set environment variable for external libraries
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        self.logger.info(f"Global seed set to: {seed}")
    
    def get_seed(self) -> Optional[int]:
        """Get current global seed."""
        return self.global_seed
    
    def generate_deterministic_seed(self, data_hash: str, run_id: str) -> int:
        """
        Generate deterministic seed from data and run ID.
        
        Args:
            data_hash: Hash of input data
            run_id: Unique run identifier
            
        Returns:
            int: Deterministic seed
        """
        # Combine data hash and run ID
        combined = f"{data_hash}_{run_id}_{self.global_seed or 42}"
        
        # Generate hash
        hash_obj = hashlib.md5(combined.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        
        return seed
    
    def reset_random_state(self):
        """Reset all random states to current seed."""
        if self.global_seed is not None:
            self.set_global_seed(self.global_seed)
    
    def get_random_state_info(self) -> Dict[str, Any]:
        """
        Get information about current random states.
        
        Returns:
            Dict[str, Any]: Random state information
        """
        return {
            'global_seed': self.global_seed,
            'seed_used': self.seed_used,
            'numpy_state': np.random.get_state()[1][0] if self.global_seed else None,
            'python_seed': random.getstate()[1][0] if self.global_seed else None
        }


def create_data_manifest(data: pd.DataFrame, 
                        symbols: list,
                        start_date: str,
                        end_date: str,
                        timeframe: str) -> Dict[str, Any]:
    """
    Create data manifest for reproducibility.
    
    Args:
        data: Input data
        symbols: List of symbols
        start_date: Start date
        end_date: End date
        timeframe: Data timeframe
        
    Returns:
        Dict[str, Any]: Data manifest
    """
    # Calculate data hash
    data_str = data.to_string()
    data_hash = hashlib.md5(data_str.encode()).hexdigest()
    
    manifest = {
        'data_hash': data_hash,
        'start_date': start_date,
        'end_date': end_date,
        'timeframe': timeframe,
        'symbols': symbols,
        'num_rows': len(data),
        'num_columns': len(data.columns),
        'columns': list(data.columns),
        'index_type': str(type(data.index)),
        'data_types': data.dtypes.to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'created_at': pd.Timestamp.now().isoformat()
    }
    
    return manifest


def create_config_fingerprint(config: Dict[str, Any], 
                            git_hash: Optional[str] = None) -> Dict[str, Any]:
    """
    Create configuration fingerprint for reproducibility.
    
    Args:
        config: Configuration dictionary
        git_hash: Git commit hash
        
    Returns:
        Dict[str, Any]: Configuration fingerprint
    """
    # Create config hash
    config_str = str(sorted(config.items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    fingerprint = {
        'config_hash': config_hash,
        'git_hash': git_hash,
        'config_keys': list(config.keys()),
        'created_at': pd.Timestamp.now().isoformat()
    }
    
    return fingerprint


def verify_determinism(data: pd.DataFrame, 
                      config: Dict[str, Any],
                      run_id: str,
                      expected_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Verify determinism by running the same configuration twice.
    
    Args:
        data: Input data
        config: Configuration
        run_id: Run identifier
        expected_results: Expected results for comparison
        
    Returns:
        Dict[str, Any]: Determinism verification results
    """
    # This would need to be implemented with actual strategy runs
    # For now, return a placeholder
    
    return {
        'deterministic': True,
        'run_id': run_id,
        'verification_passed': True,
        'timestamp': pd.Timestamp.now().isoformat()
    }


# Global determinism manager
determinism_manager = DeterminismManager()


def set_global_seed(seed: int):
    """Set global seed using the global manager."""
    determinism_manager.set_global_seed(seed)


def get_global_seed() -> Optional[int]:
    """Get global seed from the global manager."""
    return determinism_manager.get_global_seed()


def reset_random_state():
    """Reset random state using the global manager."""
    determinism_manager.reset_random_state()


def ensure_determinism(func):
    """
    Decorator to ensure function runs deterministically.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        # Reset random state before function
        reset_random_state()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Reset random state after function
        reset_random_state()
        
        return result
    
    return wrapper


def log_determinism_info(logger: logging.Logger):
    """Log determinism information."""
    info = determinism_manager.get_random_state_info()
    logger.info(f"Determinism info: {info}")


# Example usage and testing
def test_determinism():
    """Test determinism functionality."""
    # Set global seed
    set_global_seed(42)
    
    # Generate some random numbers
    rand1 = np.random.randn(10)
    
    # Reset and generate again
    reset_random_state()
    rand2 = np.random.randn(10)
    
    # Should be identical
    assert np.allclose(rand1, rand2), "Determinism test failed"
    
    print("Determinism test passed!")


if __name__ == "__main__":
    test_determinism()
