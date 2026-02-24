"""
Feature Selection and Combination module.
Analyzes feature importance and generates feature combinations for testing.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple
import json
import os


class FeatureSelector:
    """
    Analyzes feature importance and creates feature combinations for testing.
    """
    
    def __init__(self, model, feature_names: List[str], df: pd.DataFrame = None):
        """
        Initialize feature selector.
        
        Args:
            model: LightGBM model with feature importance
            feature_names: List of feature names
            df: Optional DataFrame to calculate correlations
        """
        self.model = model
        self.feature_names = np.array(feature_names)
        self.df = df
        self.feature_importance = None
        self.correlations = None
        self.combinations = {}
    
    def extract_importance(self) -> pd.DataFrame:
        """
        Extract feature importance from LightGBM model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        # LightGBM's feature_importance_
        importance_scores = self.model.feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Normalize to percentages
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        ).round(2)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        self.feature_importance = importance_df
        return importance_df
    
    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between features.
        
        Returns:
            Correlation matrix
        """
        if self.df is None:
            raise ValueError("DataFrame required to calculate correlations")
        
        self.correlations = self.df[self.feature_names].corr().abs()
        return self.correlations
    
    def find_redundant_features(self, correlation_threshold: float = 0.95) -> List[str]:
        """
        Identify redundant features with high correlation.
        
        Args:
            correlation_threshold: Correlation threshold for redundancy
            
        Returns:
            List of redundant feature names to remove
        """
        if self.correlations is None:
            self.calculate_correlations()
        
        redundant = set()
        
        for i in range(len(self.correlations.columns)):
            for j in range(i + 1, len(self.correlations.columns)):
                if self.correlations.iloc[i, j] >= correlation_threshold:
                    # Keep higher importance feature, mark lower as redundant
                    feat_i = self.correlations.columns[i]
                    feat_j = self.correlations.columns[j]
                    
                    imp_i = self.feature_importance[
                        self.feature_importance['feature'] == feat_i
                    ]['importance'].values[0]
                    imp_j = self.feature_importance[
                        self.feature_importance['feature'] == feat_j
                    ]['importance'].values[0]
                    
                    if imp_i < imp_j:
                        redundant.add(feat_i)
                    else:
                        redundant.add(feat_j)
        
        return list(redundant)
    
    def generate_combinations(
        self,
        top_k_list: List[int] = [20, 15, 10, 8, 5],
        remove_redundant: bool = True,
        correlation_threshold: float = 0.95
    ) -> Dict[str, List[str]]:
        """
        Generate multiple feature combinations for testing.
        
        Args:
            top_k_list: List of top-K values to create combinations
            remove_redundant: Whether to remove redundant features
            correlation_threshold: Correlation threshold for redundancy
            
        Returns:
            Dictionary of combination_name -> feature_list
        """
        if self.feature_importance is None:
            self.extract_importance()
        
        combinations = {}
        
        # 1. All features baseline
        combinations['all_features'] = sorted(self.feature_names.tolist())
        
        # 2. Top-K combinations
        for k in top_k_list:
            if k <= len(self.feature_names):
                top_features = self.feature_importance.head(k)['feature'].tolist()
                combinations[f'top_{k}'] = top_features
        
        # 3. Without redundant features
        if remove_redundant:
            redundant = self.find_redundant_features(correlation_threshold)
            non_redundant = [f for f in self.feature_names if f not in redundant]
            combinations['non_redundant'] = non_redundant
        
        # 4. Categorize by feature type
        combinations.update(self._get_category_combinations())
        
        self.combinations = combinations
        return combinations
    
    def _get_category_combinations(self) -> Dict[str, List[str]]:
        """
        Generate feature combinations by category/type.
        Assumes features follow naming conventions like: price_*, volume_*, momentum_*, etc.
        
        Returns:
            Dictionary of category-based combinations
        """
        category_combinations = {}
        categories = {}
        
        # Categorize features by prefix
        for feature in self.feature_names:
            # Extract prefix (first part before '_')
            prefix = feature.split('_')[0].lower()
            if prefix not in categories:
                categories[prefix] = []
            categories[prefix].append(feature)
        
        # Create combinations for each category
        for category, features in categories.items():
            if len(features) > 0:
                category_combinations[f'{category}_only'] = features
        
        # Create combination of top features per category
        top_per_category = []
        if self.feature_importance is not None:
            for category, features in categories.items():
                # Get top 2-3 features from this category
                top_in_cat = self.feature_importance[
                    self.feature_importance['feature'].isin(features)
                ].head(min(3, len(features)))['feature'].tolist()
                top_per_category.extend(top_in_cat)
            
            if top_per_category:
                category_combinations['top_per_category'] = top_per_category
        
        # Create hull + macd combination
        hull_features = categories.get('hull', [])
        macd_features = categories.get('macd', [])
        if hull_features and macd_features:
            hull_macd = hull_features + macd_features
            category_combinations['hull_macd'] = hull_macd
        
        return category_combinations
    
    def save_combinations(self, output_path: str):
        """
        Save feature combinations to JSON file.
        
        Args:
            output_path: Path to save combinations JSON
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.combinations, f, indent=2)
        
        print(f"Feature combinations saved to: {output_path}")
    
    def save_importance(self, output_path: str):
        """
        Save feature importance to CSV.
        
        Args:
            output_path: Path to save importance CSV
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.feature_importance.to_csv(output_path, index=False)
        print(f"Feature importance saved to: {output_path}")
    
    def print_summary(self):
        """Print summary of features and combinations."""
        if self.feature_importance is None:
            self.extract_importance()
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE SUMMARY")
        print("="*80)
        print(f"Total Features: {len(self.feature_names)}")
        print("\nTop 15 Features by Importance:")
        print("-"*80)
        print(self.feature_importance.head(15).to_string(index=False))
        
        if self.combinations:
            print("\n" + "="*80)
            print("FEATURE COMBINATIONS")
            print("="*80)
            for combo_name, features in self.combinations.items():
                print(f"{combo_name:<25} : {len(features):>3} features")
        
        print("="*80 + "\n")
