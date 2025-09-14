"""
Purged Cross-Validation for Time Series Data.

This module implements purged cross-validation with embargo periods to prevent
data leakage in financial time series models.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_array
import logging

logger = logging.getLogger(__name__)


class PurgedCrossValidator(BaseCrossValidator):
    """
    Purged Cross-Validator for time series data.
    
    This validator prevents data leakage by:
    1. Purging samples that overlap with the test set
    2. Adding an embargo period after the test set
    3. Ensuring no overlap between train/test sets
    """
    
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01, 
                 purge_pct: float = 0.01, min_train_size: int = 100):
        """
        Initialize the purged cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            embargo_pct: Percentage of data to embargo after test set
            purge_pct: Percentage of data to purge before test set
            min_train_size: Minimum training set size
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        self.min_train_size = min_train_size
        
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits with purging and embargo.
        
        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (not used, kept for sklearn compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        test_size = n_samples // self.n_splits
        embargo_size = int(test_size * self.embargo_pct)
        purge_size = int(test_size * self.purge_pct)
        
        for i in range(self.n_splits):
            # Calculate test set boundaries
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            
            # Create test indices
            test_indices = indices[test_start:test_end]
            
            # Calculate train set boundaries with purging and embargo
            train_end = test_start - purge_size
            train_start = max(0, train_end - test_size)
            
            # Apply embargo after test set
            embargo_start = test_end
            embargo_end = min(embargo_start + embargo_size, n_samples)
            
            # Create train indices (before test set, after purging)
            train_indices = indices[train_start:train_end]
            
            # Remove any overlap with embargo period
            train_indices = train_indices[train_indices < embargo_start]
            
            # Ensure minimum training size
            if len(train_indices) < self.min_train_size:
                logger.warning(f"Split {i}: Training set too small ({len(train_indices)} < {self.min_train_size})")
                continue
                
            # Ensure no overlap between train and test
            train_indices = train_indices[train_indices < test_start]
            
            if len(train_indices) == 0:
                logger.warning(f"Split {i}: No training samples after purging")
                continue
                
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get the number of splits."""
        return self.n_splits


class TimeSeriesSplit:
    """
    Time Series Cross-Validation with purging and embargo.
    
    This is a more flexible version that works with datetime indices.
    """
    
    def __init__(self, n_splits: int = 5, embargo_days: int = 1, 
                 purge_days: int = 1, min_train_days: int = 30):
        """
        Initialize time series splitter.
        
        Args:
            n_splits: Number of splits
            embargo_days: Days to embargo after test set
            purge_days: Days to purge before test set
            min_train_days: Minimum training period in days
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.min_train_days = min_train_days
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate time series splits with purging and embargo.
        
        Args:
            X: DataFrame with datetime index
            y: Target series (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")
            
        dates = X.index.sort_values()
        n_samples = len(dates)
        
        # Calculate split sizes
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Calculate test set boundaries
            test_start_idx = i * test_size
            test_end_idx = min((i + 1) * test_size, n_samples)
            
            test_dates = dates[test_start_idx:test_end_idx]
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            # Calculate purging period
            purge_end_date = test_dates[0] - pd.Timedelta(days=self.purge_days)
            
            # Calculate embargo period
            embargo_start_date = test_dates[-1] + pd.Timedelta(days=self.embargo_days)
            
            # Find training samples (before purge, after embargo)
            train_mask = (dates < purge_end_date) & (dates >= dates[0])
            train_indices = np.where(train_mask)[0]
            
            # Ensure minimum training period
            if len(train_indices) > 0:
                train_days = (dates[train_indices[-1]] - dates[train_indices[0]]).days
                if train_days < self.min_train_days:
                    logger.warning(f"Split {i}: Training period too short ({train_days} < {self.min_train_days} days)")
                    continue
                    
            if len(train_indices) == 0:
                logger.warning(f"Split {i}: No training samples after purging")
                continue
                
            yield train_indices, test_indices


def validate_no_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series) -> bool:
    """
    Validate that there's no data leakage between train and test sets.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        
    Returns:
        bool: True if no leakage detected
    """
    # Check temporal ordering
    if isinstance(X_train.index, pd.DatetimeIndex) and isinstance(X_test.index, pd.DatetimeIndex):
        max_train_date = X_train.index.max()
        min_test_date = X_test.index.min()
        
        if max_train_date >= min_test_date:
            logger.error(f"Temporal leakage detected: max_train_date ({max_train_date}) >= min_test_date ({min_test_date})")
            return False
    
    # Check for overlapping indices
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    
    overlap = train_indices.intersection(test_indices)
    if overlap:
        logger.error(f"Index overlap detected: {len(overlap)} overlapping indices")
        return False
        
    logger.info("No data leakage detected")
    return True


def create_embargo_mask(dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex, 
                       embargo_days: int) -> np.ndarray:
    """
    Create embargo mask for given test dates.
    
    Args:
        dates: All dates in the dataset
        test_dates: Test set dates
        embargo_days: Number of days to embargo after test set
        
    Returns:
        np.ndarray: Boolean mask for embargo period
    """
    embargo_start = test_dates.max() + pd.Timedelta(days=embargo_days)
    embargo_end = embargo_start + pd.Timedelta(days=embargo_days)
    
    return (dates >= embargo_start) & (dates < embargo_end)
