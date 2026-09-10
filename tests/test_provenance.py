"""
Tests for provenance module.
"""

import pytest
import pandas as pd

from chemdata.provenance.tracker import ProvenanceTracker


class TestProvenanceTracker:
    """Tests for ProvenanceTracker."""
    
    def test_add_provenance(self):
        """Test adding provenance information."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        tracker = ProvenanceTracker()
        result = tracker.add_provenance(
            df,
            source="DOI:10.1234/test",
            source_type="literature"
        )
        
        assert 'provenance_source' in result.columns
        assert result['provenance_source'].iloc[0] == "DOI:10.1234/test"
    
    def test_generate_provenance_report(self):
        """Test provenance report generation."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'provenance_source': ['source1', 'source1', 'source2']
        })
        
        tracker = ProvenanceTracker()
        report = tracker.generate_provenance_report(df)
        
        assert 'total_records' in report
        assert report['total_records'] == 3