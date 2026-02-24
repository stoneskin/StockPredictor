"""
Walk-Forward Validation module for time-series stock prediction.
Implements expanding windows to avoid look-ahead bias and capture market regimes.
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_size: int
    test_size: int


class WalkForwardValidator:
    """
    Implements walk-forward (expanding window) validation for time-series data.
    
    Avoids look-ahead bias by ensuring:
    - Training data always precedes test data chronologically
    - Test sets from different periods capture market regime changes
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        train_months: int = 24,
        test_months: int = 1,
        step_months: int = 3,
        min_train_samples: int = 500,
        min_test_samples: int = 20
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            df: DataFrame with datetime index or date column
            date_column: Name of date column if not using index
            train_months: Size of training window in months
            test_months: Size of test window in months
            step_months: Step size between folds in months
            min_train_samples: Minimum samples required for training
            min_test_samples: Minimum samples required for testing
        """
        self.df = df.copy()
        self.date_column = date_column
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        
        # Convert to datetime if needed
        if date_column in self.df.columns:
            self.df[date_column] = pd.to_datetime(self.df[date_column])
            self.df = self.df.sort_values(date_column).reset_index(drop=True)
            self.dates = self.df[date_column].values
        else:
            self.df = self.df.sort_index().reset_index(drop=True)
            self.dates = pd.to_datetime(self.df.index).values
        
        self.folds = None
    
    def generate_folds(self) -> List[WalkForwardPeriod]:
        """
        Generate walk-forward folds with expanding training windows.
        
        Returns:
            List of WalkForwardPeriod objects
        """
        folds = []
        fold_id = 0
        
        # Calculate window sizes in samples - use a more robust approach
        if len(self.dates) > 1:
            date_range_days = (pd.Timestamp(self.dates[-1]) - pd.Timestamp(self.dates[0])).days
            if date_range_days > 0:
                samples_per_month = len(self.df) / (date_range_days / 30)
            else:
                samples_per_month = len(self.df) / 12  # Fallback: assume 1 year of data
        else:
            samples_per_month = len(self.df) / 12  # Fallback
        
        train_samples = int(self.train_months * samples_per_month)
        test_samples = int(self.test_months * samples_per_month)
        step_samples = int(self.step_months * samples_per_month)
        
        # Ensure minimum sample sizes
        train_samples = max(train_samples, self.min_train_samples)
        test_samples = max(test_samples, self.min_test_samples)
        
        # Starting point: use enough data for initial training window
        train_start_idx = 0
        
        # Generate folds
        test_start_idx = train_samples
        
        while test_start_idx + test_samples <= len(self.df):
            train_end_idx = test_start_idx - 1
            test_end_idx = test_start_idx + test_samples - 1
            
            # Validate fold has minimum samples
            if (train_end_idx - train_start_idx + 1 >= self.min_train_samples and
                test_end_idx - test_start_idx + 1 >= self.min_test_samples):
                
                fold = WalkForwardPeriod(
                    fold_id=fold_id,
                    train_start=pd.Timestamp(self.dates[train_start_idx]).strftime('%Y-%m-%d'),
                    train_end=pd.Timestamp(self.dates[train_end_idx]).strftime('%Y-%m-%d'),
                    test_start=pd.Timestamp(self.dates[test_start_idx]).strftime('%Y-%m-%d'),
                    test_end=pd.Timestamp(self.dates[test_end_idx]).strftime('%Y-%m-%d'),
                    train_indices=np.arange(train_start_idx, train_end_idx + 1),
                    test_indices=np.arange(test_start_idx, test_end_idx + 1),
                    train_size=train_end_idx - train_start_idx + 1,
                    test_size=test_end_idx - test_start_idx + 1
                )
                
                folds.append(fold)
                fold_id += 1
            
            # Move to next test period (expanding window)
            test_start_idx += step_samples
        
        self.folds = folds
        
        # Warn if no folds were generated
        if len(folds) == 0:
            print("\nWARNING: No walk-forward folds were generated!")
            print(f"   Data size: {len(self.df)} samples")
            print(f"   Date range: {self.dates[0]} to {self.dates[-1]}")
            print(f"   Requested train window: {self.train_months} months (~{train_samples} samples)")
            print(f"   Requested test window: {self.test_months} months (~{test_samples} samples)")
            print(f"   Minimum train samples required: {self.min_train_samples}")
            print(f"   Minimum test samples required: {self.min_test_samples}")
            print("\n   Possible solutions:")
            print("   1. Reduce train_months/test_months parameters")
            print("   2. Reduce min_train_samples/min_test_samples")
            print("   3. Use smaller step_months to generate more folds")
        
        return folds
    
    def get_train_data(self, fold: WalkForwardPeriod):
        """Get training data for a fold."""
        return self.df.iloc[fold.train_indices]
    
    def get_test_data(self, fold: WalkForwardPeriod):
        """Get test data for a fold."""
        return self.df.iloc[fold.test_indices]
    
    def print_summary(self):
        """Print summary of generated folds."""
        if self.folds is None:
            self.generate_folds()
        
        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*80)
        print(f"Total Folds: {len(self.folds)}")
        print(f"Train Window: {self.train_months} months")
        print(f"Test Window: {self.test_months} months")
        print(f"Step Size: {self.step_months} months")
        print("\nFold Details:")
        print("-"*80)
        print(f"{'Fold':<6} {'Train Period':<30} {'Test Period':<30} {'Train':<8} {'Test':<8}")
        print("-"*80)
        
        for fold in self.folds:
            print(
                f"{fold.fold_id:<6} {fold.train_start} to {fold.train_end:<12} "
                f"{fold.test_start} to {fold.test_end:<12} {fold.train_size:<8} {fold.test_size:<8}"
            )
        
        print("="*80 + "\n")
    
    def __len__(self):
        """Return number of folds."""
        if self.folds is None:
            self.generate_folds()
        return len(self.folds)
    
    def __iter__(self):
        """Iterate over folds."""
        if self.folds is None:
            self.generate_folds()
        return iter(self.folds)
