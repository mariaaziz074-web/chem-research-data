"""
Tests for quality modules.
"""

import pytest
import pandas as pd
import numpy as np

from chemdata.quality.duplicates import DuplicateDetector
from chemdata.quality.outliers import OutlierDetector
from chemdata.quality.missing import MissingDataAnalyzer


class TestDuplicateDetector:
    """Tests for DuplicateDetector."""
    
    def test_find_exact_duplicates(self):
        """Test exact duplicate detection."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 1, 2],
            'B': ['a', 'b', 'c', 'a', 'b']
        })
        
        detector = DuplicateDetector()
        duplicates = detector.find_exact_duplicates(df)
        
        assert len(duplicates) == 4  # 2 pairs of duplicates
    
    def test_mark_duplicates(self):
        """Test duplicate marking."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 1],
            'B': ['a', 'b', 'c', 'a']
        })
        
        detector = DuplicateDetector()
        result = detector.mark_duplicates(df)
        
        assert 'is_duplicate' in result.columns
        assert result['is_duplicate'].sum() == 1  # One duplicate (keeping first)


class TestOutlierDetector:
    """Tests for OutlierDetector."""
    
    def test_detect_zscore(self):
        """Test Z-score outlier detection."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is clear outlier
        })
        
        detector = OutlierDetector()
        outliers = detector.detect_zscore(df, 'values', threshold=2)
        
        assert outliers.iloc[-1] == True  # Last value is outlier
    
    def test_detect_iqr(self):
        """Test IQR outlier detection."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]
        })
        
        detector = OutlierDetector()
        outliers = detector.detect_iqr(df, 'values')
        
        assert outliers.iloc[-1] == True
    
    def test_generate_outlier_report(self):
        """Test outlier report generation."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]
        })
        
        detector = OutlierDetector()
        report = detector.generate_outlier_report(df, 'values')
        
        assert 'column' in report
        assert 'methods' in report


class TestMissingDataAnalyzer:
    """Tests for MissingDataAnalyzer."""
    
    def test_count_missing(self):
        """Test missing value counting."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, None, None, 4]
        })
        
        analyzer = MissingDataAnalyzer()
        missing = analyzer.count_missing(df)
        
        assert missing['A'] == 1
        assert missing['B'] == 2
    
    def test_missing_fraction(self):
        """Test missing fraction calculation."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, None, None, 4]
        })
        
        analyzer = MissingDataAnalyzer()
        frac = analyzer.missing_fraction(df)
        
        assert abs(frac['A'] - 0.25) < 0.01
        assert abs(frac['B'] - 0.5) < 0.01
    
    def test_generate_missing_report(self):
        """Test missing data report."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [1, None, None, 4]
        })
        
        analyzer = MissingDataAnalyzer()
        report = analyzer.generate_missing_report(df)
        
        assert 'total_rows' in report
        assert 'missing_cells' in report
        assert report['missing_cells'] == 3