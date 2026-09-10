"""
Export chemistry datasets to various formats.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import json
from pathlib import Path


class DataExporter:
    """
    Export chemistry datasets to ML-ready formats.
    
    Supported formats:
    - CSV (clean)
    - JSON (with metadata)
    - Parquet (efficient)
    - Train/test splits
    
    Examples:
        >>> exporter = DataExporter()
        >>> exporter.export_csv(df, 'clean_data.csv', include_metadata=True)
        >>> exporter.export_train_test_split(df, 'train.csv', 'test.csv', test_size=0.2)
    """
    
    def __init__(self):
        """Initialize data exporter."""
        pass
    
    def export_csv(
        self,
        df: pd.DataFrame,
        output_path: str,
        include_metadata: bool = False,
        include_provenance: bool = True,
        **kwargs
    ):
        """
        Export to CSV format.
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            include_metadata: Whether to include metadata columns
            include_provenance: Whether to include provenance columns
            **kwargs: Additional arguments for pd.to_csv()
        """
        df_export = df.copy()
        
        # Optionally remove metadata columns
        if not include_metadata:
            metadata_cols = [col for col in df.columns if 'metadata' in col.lower()]
            df_export = df_export.drop(columns=metadata_cols, errors='ignore')
        
        if not include_provenance:
            prov_cols = [col for col in df.columns if col.startswith('provenance_')]
            df_export = df_export.drop(columns=prov_cols, errors='ignore')
        
        # Export
        df_export.to_csv(output_path, index=False, **kwargs)
    
    def export_json(
        self,
        df: pd.DataFrame,
        output_path: str,
        orient: str = 'records',
        include_schema: bool = True
    ):
        """
        Export to JSON format.
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            orient: JSON orientation ('records', 'index', 'columns')
            include_schema: Whether to include schema information
        """
        data = {
            'data': json.loads(df.to_json(orient=orient)),
            'metadata': {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'columns': list(df.columns)
            }
        }
        
        if include_schema:
            data['schema'] = {
                col: str(dtype) for col, dtype in df.dtypes.items()
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_parquet(
        self,
        df: pd.DataFrame,
        output_path: str,
        **kwargs
    ):
        """
        Export to Parquet format (efficient for large datasets).
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            **kwargs: Additional arguments for pd.to_parquet()
        """
        df.to_parquet(output_path, index=False, **kwargs)
    
    def export_train_test_split(
        self,
        df: pd.DataFrame,
        train_path: str,
        test_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: Optional[str] = None
    ):
        """
        Export train/test split.
        
        Args:
            df: DataFrame to split
            train_path: Training set output path
            test_path: Test set output path
            test_size: Fraction for test set (default: 0.2)
            random_state: Random seed
            stratify_column: Column to stratify by (optional)
        """
        from sklearn.model_selection import train_test_split
        
        # Prepare stratify
        stratify = df[stratify_column] if stratify_column else None
        
        # Split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Export
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Export metadata
        metadata = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'test_fraction': test_size,
            'random_state': random_state,
            'stratify_column': stratify_column
        }
        
        metadata_path = Path(train_path).parent / 'split_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def export_ml_ready(
        self,
        df: pd.DataFrame,
        output_dir: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        format: str = 'csv'
    ):
        """
        Export ML-ready dataset with train/test split and feature separation.
        
        Args:
            df: DataFrame to export
            output_dir: Output directory
            target_column: Target variable column
            feature_columns: Feature columns (None = all except target)
            test_size: Test set fraction
            format: Export format ('csv', 'parquet')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42
        )
        
        # Export train
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        
        # Export test
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]
        
        # Save files
        if format == 'csv':
            X_train.to_csv(output_dir / 'X_train.csv', index=False)
            y_train.to_csv(output_dir / 'y_train.csv', index=False, header=True)
            X_test.to_csv(output_dir / 'X_test.csv', index=False)
            y_test.to_csv(output_dir / 'y_test.csv', index=False, header=True)
        elif format == 'parquet':
            X_train.to_parquet(output_dir / 'X_train.parquet', index=False)
            y_train.to_frame().to_parquet(output_dir / 'y_train.parquet', index=False)
            X_test.to_parquet(output_dir / 'X_test.parquet', index=False)
            y_test.to_frame().to_parquet(output_dir / 'y_test.parquet', index=False)
        
        # Export metadata
        metadata = {
            'target_column': target_column,
            'feature_columns': feature_columns,
            'n_features': len(feature_columns),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'test_fraction': test_size,
            'format': format
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)