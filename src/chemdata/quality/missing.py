"""
Missing data analysis for chemistry datasets.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class MissingDataAnalyzer:
    """
    Analyze missing data patterns in chemistry datasets.
    
    Identifies:
    - Missing data patterns (MCAR, MAR, MNAR)
    - Columns with high missingness
    - Correlations in missingness
    
    Examples:
        >>> analyzer = MissingDataAnalyzer()
        >>> report = analyzer.generate_missing_report(df)
        >>> pattern = analyzer.analyze_missing_pattern(df)
    """
    
    def __init__(self):
        """Initialize missing data analyzer."""
        pass
    
    def count_missing(self, df: pd.DataFrame) -> pd.Series:
        """
        Count missing values per column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with missing counts per column
        """
        return df.isna().sum()
    
    def missing_fraction(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate fraction of missing values per column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with missing fractions per column
        """
        return df.isna().sum() / len(df)
    
    def missing_by_row(self, df: pd.DataFrame) -> pd.Series:
        """
        Count missing values per row.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Series with missing counts per row
        """
        return df.isna().sum(axis=1)
    
    def identify_high_missing_columns(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> List[str]:
        """
        Identify columns with high missingness.
        
        Args:
            df: Input DataFrame
            threshold: Missing fraction threshold (default: 0.5)
            
        Returns:
            List of column names with high missingness
        """
        missing_frac = self.missing_fraction(df)
        high_missing = missing_frac[missing_frac > threshold]
        return high_missing.index.tolist()
    
    def analyze_missing_pattern(self, df: pd.DataFrame) -> Dict:
        """
        Analyze patterns of missingness.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with pattern analysis
        """
        # Create binary missing indicator
        missing_indicator = df.isna().astype(int)
        
        # Count unique missing patterns
        pattern_counts = missing_indicator.value_counts()
        
        # Most common patterns
        top_patterns = pattern_counts.head(10)
        
        report = {
            'total_patterns': len(pattern_counts),
            'rows_with_no_missing': int((missing_indicator.sum(axis=1) == 0).sum()),
            'rows_with_any_missing': int((missing_indicator.sum(axis=1) > 0).sum()),
            'rows_with_all_missing': int((missing_indicator.sum(axis=1) == len(df.columns)).sum()),
            'top_patterns': top_patterns.to_dict()
        }
        
        return report
    
    def missing_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation between missing values in different columns.
        
        High correlation suggests MAR (Missing At Random) rather than MCAR.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Correlation matrix of missing indicators
        """
        missing_indicator = df.isna().astype(int)
        
        # Only include columns with some missing data
        cols_with_missing = missing_indicator.columns[missing_indicator.sum() > 0]
        
        if len(cols_with_missing) == 0:
            return pd.DataFrame()
        
        missing_corr = missing_indicator[cols_with_missing].corr()
        
        return missing_corr
    
    def suggest_imputation_strategy(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Dict:
        """
        Suggest imputation strategy for a column.
        
        Args:
            df: Input DataFrame
            column: Column to analyze
            
        Returns:
            Dictionary with suggestions
        """
        missing_frac = df[column].isna().sum() / len(df)
        
        suggestion = {
            'column': column,
            'missing_fraction': float(missing_frac),
            'strategy': None,
            'reasoning': None
        }
        
        if missing_frac == 0:
            suggestion['strategy'] = 'none'
            suggestion['reasoning'] = 'No missing values'
        
        elif missing_frac < 0.05:
            suggestion['strategy'] = 'drop_rows'
            suggestion['reasoning'] = 'Very low missingness (<5%), safe to drop rows'
        
        elif missing_frac < 0.3:
            if pd.api.types.is_numeric_dtype(df[column]):
                suggestion['strategy'] = 'mean_median_imputation'
                suggestion['reasoning'] = 'Moderate missingness, numeric column'
            else:
                suggestion['strategy'] = 'mode_imputation'
                suggestion['reasoning'] = 'Moderate missingness, categorical column'
        
        elif missing_frac < 0.7:
            suggestion['strategy'] = 'model_based_imputation'
            suggestion['reasoning'] = 'High missingness, use other columns to predict'
        
        else:
            suggestion['strategy'] = 'drop_column'
            suggestion['reasoning'] = 'Very high missingness (>70%), consider dropping column'
        
        return suggestion
    
    def generate_missing_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive missing data report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Report dictionary
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'missing_cells': int(df.isna().sum().sum()),
            'missing_fraction': float(df.isna().sum().sum()) / (len(df) * len(df.columns)),
            'columns': {},
            'rows_analysis': {},
            'patterns': None,
            'recommendations': []
        }
        
        # Per-column analysis
        for col in df.columns:
            n_missing = df[col].isna().sum()
            frac_missing = n_missing / len(df)
            
            report['columns'][col] = {
                'missing_count': int(n_missing),
                'missing_fraction': float(frac_missing),
                'dtype': str(df[col].dtype)
            }
            
            # Add suggestion
            if n_missing > 0:
                suggestion = self.suggest_imputation_strategy(df, col)
                report['columns'][col]['suggestion'] = suggestion['strategy']
        
        # Row analysis
        missing_per_row = self.missing_by_row(df)
        report['rows_analysis'] = {
            'rows_complete': int((missing_per_row == 0).sum()),
            'rows_with_missing': int((missing_per_row > 0).sum()),
            'max_missing_in_row': int(missing_per_row.max()),
            'mean_missing_per_row': float(missing_per_row.mean())
        }
        
        # Pattern analysis
        report['patterns'] = self.analyze_missing_pattern(df)
        
        # Overall recommendations
        high_missing_cols = self.identify_high_missing_columns(df, threshold=0.7)
        if high_missing_cols:
            report['recommendations'].append(
                f"Consider dropping columns with >70% missing: {high_missing_cols}"
            )
        
        if report['rows_analysis']['rows_with_missing'] / len(df) > 0.5:
            report['recommendations'].append(
                "More than 50% of rows have missing data - careful with imputation"
            )
        
        return report
    
    def visualize_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary table for visualization.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Summary DataFrame
        """
        summary = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isna().sum().values,
            'Missing_Fraction': (df.isna().sum() / len(df)).values,
            'Data_Type': df.dtypes.values
        })
        
        summary = summary.sort_values('Missing_Fraction', ascending=False)
        summary['Missing_Percent'] = summary['Missing_Fraction'] * 100
        
        return summary