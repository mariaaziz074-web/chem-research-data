"""
Tests for normalization modules.
"""

import pytest
import pandas as pd
import numpy as np

from chemdata.normalization.units import UnitNormalizer
from chemdata.normalization.chemicals import ChemicalNormalizer
from chemdata.normalization.conditions import ConditionNormalizer


class TestUnitNormalizer:
    """Tests for UnitNormalizer."""
    
    def test_convert_temperature(self):
        """Test temperature conversion."""
        normalizer = UnitNormalizer()
        
        # C to K
        result = normalizer.convert_temperature(25, 'C', 'K')
        assert abs(result - 298.15) < 0.01
        
        # K to C
        result = normalizer.convert_temperature(298.15, 'K', 'C')
        assert abs(result - 25) < 0.01
    
    def test_convert_time(self):
        """Test time conversion."""
        normalizer = UnitNormalizer()
        
        # minutes to seconds
        result = normalizer.convert_time(2, 'min', 's')
        assert abs(result - 120) < 0.01
        
        # hours to minutes
        result = normalizer.convert_time(1.5, 'h', 'min')
        assert abs(result - 90) < 0.01
    
    def test_convert_energy(self):
        """Test energy conversion."""
        normalizer = UnitNormalizer()
        
        # eV to kcal/mol
        result = normalizer.convert_energy(1, 'eV', 'kcal/mol')
        assert abs(result - 23.06) < 0.1
        
        # Same unit
        result = normalizer.convert_energy(5, 'eV', 'eV')
        assert result == 5
    
    def test_convert_concentration(self):
        """Test concentration conversion."""
        normalizer = UnitNormalizer()
        
        # mg/L to mM (need molecular weight)
        result = normalizer.convert_concentration(
            100, 'mg/L', 'mM', molecular_weight=100
        )
        assert abs(result - 1.0) < 0.01
        
        # mM to mg/L
        result = normalizer.convert_concentration(
            1.0, 'mM', 'mg/L', molecular_weight=100
        )
        assert abs(result - 100) < 0.01


class TestChemicalNormalizer:
    """Tests for ChemicalNormalizer."""
    
    def test_normalize_formula(self):
        """Test formula normalization."""
        normalizer = ChemicalNormalizer()
        
        assert normalizer.normalize_formula("Ti O2") == "TiO2"
        assert normalizer.normalize_formula("TiO₂") == "TiO2"
        assert normalizer.normalize_formula("Fe2O3") == "Fe2O3"
    
    def test_standardize_smiles(self):
        """Test SMILES standardization."""
        normalizer = ChemicalNormalizer()
        
        # Canonical SMILES
        result = normalizer.standardize_smiles("CCO")
        if result:  # Only if RDKit available
            assert result == "CCO"
    
    def test_normalize_catalyst_name(self):
        """Test catalyst name normalization."""
        normalizer = ChemicalNormalizer()
        
        assert "P25" in normalizer.normalize_catalyst_name("Degussa P25")
        assert normalizer.normalize_catalyst_name("TiO2") == "TiO2"


class TestConditionNormalizer:
    """Tests for ConditionNormalizer."""
    
    def test_normalize_ph(self):
        """Test pH normalization."""
        normalizer = ConditionNormalizer()
        
        assert normalizer.normalize_ph("7.0") == 7.0
        assert normalizer.normalize_ph(7) == 7.0
        assert normalizer.normalize_ph("pH 6.5") == 6.5
        assert normalizer.normalize_ph("neutral") == 7.0
    
    def test_normalize_light_source(self):
        """Test light source normalization."""
        normalizer = ConditionNormalizer()
        
        assert normalizer.normalize_light_source("UV") == "UV"
        assert normalizer.normalize_light_source("ultraviolet") == "UV"
        assert normalizer.normalize_light_source("visible") == "Visible"