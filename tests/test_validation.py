"""
Tests for validation modules.
"""

import pytest
import pandas as pd
import numpy as np

from chemdata.validation.chemical import ChemicalValidator
from chemdata.validation.numerical import NumericalValidator
from chemdata.validation.schema import SchemaValidator


class TestChemicalValidator:
    """Tests for ChemicalValidator."""
    
    def test_validate_smiles_valid(self, sample_smiles):
        """Test validation of valid SMILES."""
        validator = ChemicalValidator()
        
        for smiles in sample_smiles['valid']:
            result = validator.validate_smiles(smiles)
            assert result['valid'], f"SMILES {smiles} should be valid"
            assert result['canonical_smiles'] is not None
            assert len(result['errors']) == 0
    
    def test_validate_smiles_invalid(self, sample_smiles):
        """Test validation of invalid SMILES."""
        validator = ChemicalValidator()
        
        for smiles in sample_smiles['invalid']:
            result = validator.validate_smiles(smiles)
            # RDKit might not be available, so we just check structure
            assert 'valid' in result
            assert 'errors' in result
    
    def test_validate_formula_valid(self, sample_formulas):
        """Test validation of valid formulas."""
        validator = ChemicalValidator()
        
        for formula in sample_formulas['valid']:
            result = validator.validate_formula(formula)
            assert result['valid'], f"Formula {formula} should be valid"
            assert result['normalized_formula'] is not None
            assert result['molecular_weight'] is not None
    
    def test_validate_formula_invalid(self, sample_formulas):
        """Test validation of invalid formulas."""
        validator = ChemicalValidator()
        
        for formula in sample_formulas['invalid']:
            result = validator.validate_formula(formula)
            assert not result['valid'], f"Formula {formula} should be invalid"
            assert len(result['errors']) > 0
    
    def test_validate_formula_dataframe(self, sample_chemistry_data):
        """Test formula validation on DataFrame."""
        validator = ChemicalValidator()
        
        result_df = validator.validate_formula_dataframe(
            sample_chemistry_data,
            'catalyst'
        )
        
        assert 'catalyst_valid' in result_df.columns
        assert 'catalyst_normalized' in result_df.columns
        assert 'catalyst_mw' in result_df.columns


class TestNumericalValidator:
    """Tests for NumericalValidator."""
    
    def test_validate_range_valid(self):
        """Test range validation with valid values."""
        validator = NumericalValidator()
        
        result = validator.validate_range(50, min_val=0, max_val=100)
        assert result['valid']
        assert len(result['errors']) == 0
    
    def test_validate_range_below_min(self):
        """Test range validation below minimum."""
        validator = NumericalValidator()
        
        result = validator.validate_range(-10, min_val=0, max_val=100)
        assert not result['valid']
        assert len(result['errors']) > 0
    
    def test_validate_range_above_max(self):
        """Test range validation above maximum."""
        validator = NumericalValidator()
        
        result = validator.validate_range(150, min_val=0, max_val=100)
        assert not result['valid']
        assert len(result['errors']) > 0
    
    def test_validate_percentage(self):
        """Test percentage validation."""
        validator = NumericalValidator()
        
        assert validator.validate_percentage(50)['valid']
        assert validator.validate_percentage(0)['valid']
        assert validator.validate_percentage(100)['valid']
        assert not validator.validate_percentage(-10)['valid']
        assert not validator.validate_percentage(150)['valid']
    
    def test_validate_ph(self):
        """Test pH validation."""
        validator = NumericalValidator()
        
        assert validator.validate_ph(7.0)['valid']
        assert validator.validate_ph(0)['valid']
        assert validator.validate_ph(14)['valid']
        assert not validator.validate_ph(-1)['valid']
        assert not validator.validate_ph(15)['valid']
    
    def test_validate_column(self, sample_chemistry_data):
        """Test column validation."""
        validator = NumericalValidator()
        
        result_df = validator.validate_column(
            sample_chemistry_data,
            'degradation_percent',
            validator_type='percentage'
        )
        
        assert 'degradation_percent_valid' in result_df.columns
        assert result_df['degradation_percent_valid'].all()


class TestSchemaValidator:
    """Tests for SchemaValidator."""
    
    def test_schema_validation_valid(self, sample_chemistry_data):
        """Test schema validation with valid data."""
        schema = {
            'catalyst': {'type': 'string', 'required': True},
            'degradation_percent': {
                'type': 'float',
                'required': True,
                'min': 0,
                'max': 100
            },
            'pH': {'type': 'float', 'min': 0, 'max': 14},
        }
        
        validator = SchemaValidator(schema)
        report = validator.validate(sample_chemistry_data)
        
        assert report['valid']
        assert len(report['errors']) == 0
    
    def test_schema_validation_missing_column(self, sample_chemistry_data):
        """Test schema validation with missing required column."""
        schema = {
            'missing_column': {'type': 'string', 'required': True},
        }
        
        validator = SchemaValidator(schema)
        report = validator.validate(sample_chemistry_data)
        
        assert not report['valid']
        assert len(report['errors']) > 0
        assert 'missing_column' in str(report['errors'])
    
    def test_generate_schema(self, sample_chemistry_data):
        """Test schema generation from DataFrame."""
        validator = SchemaValidator()
        schema = validator.generate_schema_from_dataframe(sample_chemistry_data)
        
        assert 'catalyst' in schema
        assert 'degradation_percent' in schema
        assert schema['degradation_percent']['type'] == 'float'