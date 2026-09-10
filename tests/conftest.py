"""
Pytest configuration and fixtures.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_chemistry_data():
    """Sample chemistry dataset for testing."""
    data = {
        'catalyst': ['TiO2', 'ZnO', 'Fe2O3', 'CuO', 'NiO'],
        'catalyst_loading_g_L': [1.0, 0.5, 2.0, 1.5, 1.0],
        'dye': ['Methylene Blue', 'Rhodamine B', 'Methyl Orange', 'Congo Red', 'Malachite Green'],
        'dye_smiles': ['CN(C)C1=CC2=C(C=C1)SC3=CC(=[N+](C)C)C=CC3=N2', 'CC', 'O=C(O)c1ccccc1', None, 'invalid_smiles'],
        'concentration_mg_L': [10.0, 20.0, 15.0, 25.0, 10.0],
        'pH': [7.0, 6.5, 8.0, 7.5, 6.0],
        'temperature_C': [25.0, 30.0, 25.0, 35.0, 25.0],
        'time_min': [60, 120, 90, 60, 180],
        'degradation_percent': [85.0, 92.0, 78.0, 88.0, 95.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_formulas():
    """Sample molecular formulas for testing."""
    return {
        'valid': ['TiO2', 'Fe2O3', 'CuO', 'H2O', 'C6H12O6', 'NaCl'],
        'invalid': ['XYZ', '123', '', 'Ti O 2 (malformed)', 'H2O@'],
    }


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return {
        'valid': ['CCO', 'c1ccccc1', 'CC(=O)O', 'C1=CC=CC=C1'],
        'invalid': ['invalid', 'C1CCC', '', 'xyz123'],
    }