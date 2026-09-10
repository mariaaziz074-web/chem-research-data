"""
Outlier detection for chemistry datasets.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class OutlierDetector:
    """
    Detect outliers in chemistry datasets.
    
    Methods:
    - Z-score (statistical)
    - IQR (interquartile range)
    - Isolation Forest (ML-based)
    - Domain-specific rules
    
    Examples:
        >>> detector = OutlierDetector()
        >>> outliers = detector.detect_zscore(df, 'degradation_percent', threshold=3)
        >>> df_marked = detector.mark_outliers(df, 'temperature', methods=['zscore', 'iqr'])
    """
    
    def __init__(self):
        """Initialize outlier detector."""
        pass
    
    def detect_zscore(
        self,
        df: pd.DataFrame,
        column: str,
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            df: Input DataFrame
            column: Column to check
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Boolean Series indicating outliers
        """
        values = df[column].dropna()
        
        if len(values) == 0:
            return pd.Series([False] * len(df), index=df.index)
        
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            return pd.Series([False] * len(df), index=df.index)
        
        z_scores = np.abs((df[column] - mean) / std)
        outliers = z_scores > threshold
        
        return outliers.fillna(False)
    
    def detect_iqr(
        self,
        df: pd.DataFrame,
        column: str,
        multiplier: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers using IQR (Interquartile Range) method.
        
        Args:
            df: Input DataFrame
            column: Column to check
            multiplier: IQR multiplier (default: 1.5)
            
        Returns:
            Boolean Series indicating outliers
        """
        values = df[column].dropna()
        
        if len(values) == 0:
            return pd.Series([False] * len(df), index=df.index)
        
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        return outliers.fillna(False)
    
    def detect_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: List[str],
        contamination: float = 0.1,
        random_state: int = 42
    ) -> pd.Series:
        """
        Detect outliers using Isolation Forest (ML method).
        
        Args:
            df: Input DataFrame
            columns: Columns to use for detection
            contamination: Expected proportion of outliers (default: 0.1)
            random_state: Random seed
            
        Returns:
            Boolean Series indicating outliers
        """
        # Prepare data
        data = df[columns].copy()
        
        # Handle missing values
        data_clean = data.dropna()
        
        if len(data_clean) < 10:
            return pd.Series([False] * len(df), index=df.index)
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        predictions = iso_forest.fit_predict(data_scaled)
        
        # -1 indicates outlier, 1 indicates inlier
        outliers_clean = predictions == -1
        
        # Map back to original DataFrame
        outliers = pd.Series([False] * len(df), index=df.index)
        outliers.loc[data_clean.index] = outliers_clean
        
        return outliers
    
    def detect_domain_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        domain_rules: Dict[str, Tuple[float, float]]
    ) -> pd.Series:
        """
        Detect outliers based on domain-specific rules.
        
        Args:
            df: Input DataFrame
            column: Column to check
            domain_rules: Dictionary mapping column to (min, max) valid range
            
        Returns:
            Boolean Series indicating outliers
        """
        if column not in domain_rules:
            return pd.Series([False] * len(df), index=df.index)
        
        min_val, max_val = domain_rules[column]
        
        outliers = pd.Series([False] * len(df), index=df.index)
        
        if min_val is not None:
            outliers |= (df[column] < min_val)
        
        if max_val is not None:
            outliers |= (df[column] > max_val)
        
        return outliers.fillna(False)
    
    def mark_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        methods: List[str] = ['zscore', 'iqr'],
        **kwargs
    ) -> pd.DataFrame:
        """
        Mark outliers using multiple methods.
        
        Args:
            df: Input DataFrame
            column: Column to check
            methods: List of methods to use
            **kwargs: Additional arguments for specific methods
            
        Returns:
            DataFrame with outlier columns added
        """
        df = df.copy()
        
        for method in methods:
            if method == 'zscore':
                threshold = kwargs.get('zscore_threshold', 3.0)
                df[f'{column}_outlier_zscore'] = self.detect_zscore(
                    df, column, threshold
                )
            
            elif method == 'iqr':
                multiplier = kwargs.get('iqr_multiplier', 1.5)
                df[f'{column}_outlier_iqr'] = self.detect_iqr(
                    df, column, multiplier
                )
            
            elif method == 'isolation_forest':
                contamination = kwargs.get('contamination', 0.1)
                columns = kwargs.get('columns', [column])
                df[f'{column}_outlier_iforest'] = self.detect_isolation_forest(
                    df, columns, contamination
                )
            
            elif method == 'domain':
                domain_rules = kwargs.get('domain_rules', {})
                df[f'{column}_outlier_domain'] = self.detect_domain_outliers(
                    df, column, domain_rules
                )
        
        # Create consensus column (outlier if flagged by any method)
        outlier_cols = [col for col in df.columns if col.startswith(f'{column}_outlier_')]
        if outlier_cols:
            df[f'{column}_outlier_any'] = df[outlier_cols].any(axis=1)
        
        return df
    
    def generate_outlier_report(
        self,
        df: pd.DataFrame,
        column: str,
        methods: List[str] = ['zscore', 'iqr']
    ) -> Dict:
        """
        Generate outlier detection report.
        
        Args:
            df: Input DataFrame
            column: Column to analyze
            methods: Detection methods to use
            
        Returns:
            Report dictionary
        """
        report = {
            'column': column,
            'total_values': len(df[column].dropna()),
            'methods': {},
        }
        
        for method in methods:
            if method == 'zscore':
                outliers = self.detect_zscore(df, column)
                report['methods']['zscore'] = {
                    'n_outliers': int(outliers.sum()),
                    'fraction': float(outliers.sum()) / len(df)
                }
            
            elif method == 'iqr':
                outliers = self.detect_iqr(df, column)
                report['methods']['iqr'] = {
                    'n_outliers': int(outliers.sum()),
                    'fraction': float(outliers.sum()) / len(df)
                }
        
        return report
    
    def get_outlier_details(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Get details of detected outliers.
        
        Args:
            df: Input DataFrame
            column: Column analyzed
            method: Detection method used
            
        Returns:
            DataFrame with outlier rows and details
        """
        if method == 'zscore':
            outliers = self.detect_zscore(df, column)
            values = df[column]
            mean = values.mean()
            std = values.std()
            z_scores = (values - mean) / std if std > 0 else pd.Series([0] * len(values))
            
            outlier_df = df[outliers].copy()
            outlier_df[f'{column}_zscore'] = z_scores[outliers]
            outlier_df[f'{column}_deviation'] = np.abs(values[outliers] - mean)
            
            return outlier_df
        
        elif method == 'iqr':
            outliers = self.detect_iqr(df, column)
            return df[outliers].copy()
        
        return pd.DataFrame()