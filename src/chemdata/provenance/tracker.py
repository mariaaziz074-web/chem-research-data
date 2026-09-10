"""
Data provenance tracking system.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import json


class ProvenanceTracker:
    """
    Track data provenance and transformations.
    
    Examples:
        >>> tracker = ProvenanceTracker()
        >>> df_tracked = tracker.add_provenance(
        ...     df,
        ...     source="DOI:10.1234/example",
        ...     source_type="literature"
        ... )
    """
    
    def __init__(self):
        """Initialize provenance tracker."""
        self.transformations = []
    
    def add_provenance(
        self,
        df: pd.DataFrame,
        source: str,
        source_type: str = "unknown",
        source_location: Optional[str] = None,
        extraction_date: Optional[str] = None,
        quality_flags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add provenance information to DataFrame.
        
        Args:
            df: Input DataFrame
            source: Source identifier (DOI, citation, file path)
            source_type: Type of source ('literature', 'experiment', 'calculation', 'database')
            source_location: Location in source (e.g., "Table 2, page 5")
            extraction_date: Date of extraction (ISO format)
            quality_flags: List of quality flags (e.g., ['verified', 'high_confidence'])
            metadata: Additional metadata dictionary
            
        Returns:
            DataFrame with provenance columns added
        """
        df = df.copy()
        
        if extraction_date is None:
            extraction_date = datetime.now().isoformat()
        
        # Add provenance columns
        df['provenance_source'] = source
        df['provenance_source_type'] = source_type
        df['provenance_location'] = source_location
        df['provenance_extraction_date'] = extraction_date
        
        if quality_flags:
            df['provenance_quality_flags'] = [quality_flags] * len(df)
        
        if metadata:
            df['provenance_metadata'] = [metadata] * len(df)
        
        return df
    
    def log_transformation(
        self,
        df: pd.DataFrame,
        transformation_type: str,
        details: Dict[str, Any],
        columns_affected: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Log a transformation applied to the dataset.
        
        Args:
            df: DataFrame
            transformation_type: Type of transformation ('normalization', 'validation', 'cleaning')
            details: Details of transformation
            columns_affected: List of columns affected
            
        Returns:
            DataFrame with transformation logged
        """
        transformation_record = {
            'timestamp': datetime.now().isoformat(),
            'type': transformation_type,
            'details': details,
            'columns_affected': columns_affected or [],
        }
        
        self.transformations.append(transformation_record)
        
        # Optionally add to DataFrame
        if 'provenance_transformations' not in df.columns:
            df['provenance_transformations'] = [[] for _ in range(len(df))]
        
        # Add transformation to all rows
        df['provenance_transformations'] = df['provenance_transformations'].apply(
            lambda x: x + [transformation_record] if isinstance(x, list) else [transformation_record]
        )
        
        return df
    
    def generate_provenance_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate provenance report for dataset.
        
        Args:
            df: DataFrame with provenance information
            
        Returns:
            Provenance report dictionary
        """
        report = {
            'total_records': len(df),
            'sources': {},
            'transformations': self.transformations,
            'generation_date': datetime.now().isoformat(),
        }
        
        # Summarize sources
        if 'provenance_source' in df.columns:
            source_counts = df['provenance_source'].value_counts().to_dict()
            report['sources'] = source_counts
        
        # Summarize source types
        if 'provenance_source_type' in df.columns:
            type_counts = df['provenance_source_type'].value_counts().to_dict()
            report['source_types'] = type_counts
        
        return report
    
    def export_provenance(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'json'
    ):
        """
        Export provenance information.
        
        Args:
            df: DataFrame with provenance
            output_path: Output file path
            format: Export format ('json', 'csv')
        """
        report = self.generate_provenance_report(df)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        elif format == 'csv':
            # Export provenance columns only
            provenance_cols = [col for col in df.columns if col.startswith('provenance_')]
            if provenance_cols:
                df[provenance_cols].to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")