"""
Chemical structure validation.

Validates SMILES strings, molecular formulas, and chemical plausibility.
"""

from typing import Dict, List, Optional, Tuple
import re
import warnings

import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Chemical validation will be limited.")


class ChemicalValidator:
    """
    Validator for chemical structures and formulas.
    
    Examples:
        >>> validator = ChemicalValidator()
        >>> validator.validate_smiles("CCO")
        {'valid': True, 'canonical_smiles': 'CCO', 'errors': []}
        
        >>> validator.validate_formula("TiO2")
        {'valid': True, 'normalized_formula': 'TiO2', 'molecular_weight': 79.87, 'errors': []}
    """
    
    def __init__(self):
        """Initialize chemical validator."""
        self.rdkit_available = RDKIT_AVAILABLE
        
    def validate_smiles(self, smiles: str) -> Dict:
        """
        Validate SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            Dictionary with validation results:
                - valid: bool
                - canonical_smiles: str (if valid)
                - errors: list of error messages
                - warnings: list of warning messages
        """
        result = {
            'valid': False,
            'canonical_smiles': None,
            'errors': [],
            'warnings': []
        }
        
        # Check if SMILES is string
        if not isinstance(smiles, str):
            result['errors'].append(f"SMILES must be string, got {type(smiles)}")
            return result
            
        # Check if empty
        if not smiles.strip():
            result['errors'].append("SMILES string is empty")
            return result
            
        if not self.rdkit_available:
            result['warnings'].append("RDKit not available, validation limited")
            result['valid'] = True
            result['canonical_smiles'] = smiles
            return result
            
        # Parse with RDKit
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result['errors'].append("Invalid SMILES: RDKit cannot parse")
                return result
                
            # Get canonical SMILES
            canonical = Chem.MolToSmiles(mol, canonical=True)
            
            # Basic chemical plausibility checks
            warnings_list = self._check_molecular_plausibility(mol)
            
            result['valid'] = True
            result['canonical_smiles'] = canonical
            result['warnings'].extend(warnings_list)
            
        except Exception as e:
            result['errors'].append(f"Error parsing SMILES: {str(e)}")
            
        return result
    
    def validate_formula(self, formula: str) -> Dict:
        """
        Validate molecular formula.
        
        Args:
            formula: Molecular formula (e.g., "TiO2", "C6H12O6")
            
        Returns:
            Dictionary with validation results:
                - valid: bool
                - normalized_formula: str
                - molecular_weight: float (if calculable)
                - errors: list of error messages
        """
        result = {
            'valid': False,
            'normalized_formula': None,
            'molecular_weight': None,
            'errors': []
        }
        
        if not isinstance(formula, str):
            result['errors'].append(f"Formula must be string, got {type(formula)}")
            return result
            
        if not formula.strip():
            result['errors'].append("Formula is empty")
            return result
            
        # Normalize formula (remove spaces, subscript numbers, etc.)
        normalized = self._normalize_formula(formula)
        
        # Check formula pattern
        if not self._is_valid_formula_pattern(normalized):
            result['errors'].append(f"Invalid formula pattern: {formula}")
            return result
            
        # Try to calculate molecular weight
        try:
            mw = self._calculate_molecular_weight(normalized)
            result['molecular_weight'] = round(mw, 2)
        except Exception as e:
            result['errors'].append(f"Cannot calculate molecular weight: {str(e)}")
            return result
            
        result['valid'] = True
        result['normalized_formula'] = normalized
        
        return result
    
    def validate_formula_dataframe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Validate formulas in a DataFrame column.
        
        Args:
            df: Input DataFrame
            column: Column name containing formulas
            
        Returns:
            DataFrame with validation results added
        """
        validation_results = df[column].apply(self.validate_formula)
        
        df[f'{column}_valid'] = validation_results.apply(lambda x: x['valid'])
        df[f'{column}_normalized'] = validation_results.apply(
            lambda x: x.get('normalized_formula')
        )
        df[f'{column}_mw'] = validation_results.apply(
            lambda x: x.get('molecular_weight')
        )
        df[f'{column}_errors'] = validation_results.apply(
            lambda x: '; '.join(x['errors']) if x['errors'] else None
        )
        
        return df
    
    def validate_smiles_dataframe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Validate SMILES in a DataFrame column.
        
        Args:
            df: Input DataFrame
            column: Column name containing SMILES
            
        Returns:
            DataFrame with validation results added
        """
        validation_results = df[column].apply(self.validate_smiles)
        
        df[f'{column}_valid'] = validation_results.apply(lambda x: x['valid'])
        df[f'{column}_canonical'] = validation_results.apply(
            lambda x: x.get('canonical_smiles')
        )
        df[f'{column}_errors'] = validation_results.apply(
            lambda x: '; '.join(x['errors']) if x['errors'] else None
        )
        df[f'{column}_warnings'] = validation_results.apply(
            lambda x: '; '.join(x['warnings']) if x['warnings'] else None
        )
        
        return df
    
    @staticmethod
    def _normalize_formula(formula: str) -> str:
        """Normalize chemical formula."""
        # Remove spaces
        normalized = formula.replace(' ', '')
        
        # Convert subscript numbers to normal numbers
        subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        normalized = normalized.translate(subscript_map)
        
        # Convert superscript to nothing (charge indicators)
        superscript_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')
        normalized = normalized.translate(superscript_map)
        
        return normalized
    
    @staticmethod
    def _is_valid_formula_pattern(formula: str) -> bool:
        """Check if formula matches valid pattern."""
        # Pattern: Element (capital + optional lowercase) followed by optional number
        # Allows parentheses for complex formulas
        pattern = r'^([A-Z][a-z]?\d*|\(|\))+$'
        return bool(re.match(pattern, formula))
    
    @staticmethod
    def _calculate_molecular_weight(formula: str) -> float:
        """
        Calculate molecular weight from formula.
        
        Simple implementation for common elements.
        """
        # Atomic weights for common elements (simplified)
        atomic_weights = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
            'Ti': 47.867, 'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38,
            'Br': 79.904, 'I': 126.90, 'Ag': 107.87, 'Au': 196.97,
            'Na': 22.990, 'K': 39.098, 'Ca': 40.078, 'Mg': 24.305,
            'Al': 26.982, 'Si': 28.086, 'Ni': 58.693, 'Pt': 195.08,
            'Pd': 106.42, 'Co': 58.933, 'Mn': 54.938, 'Cr': 51.996,
        }
        
        # Parse formula
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        total_weight = 0.0
        for element, count in matches:
            if element not in atomic_weights:
                raise ValueError(f"Unknown element: {element}")
            count = int(count) if count else 1
            total_weight += atomic_weights[element] * count
            
        return total_weight
    
    @staticmethod
    def _check_molecular_plausibility(mol) -> List[str]:
        """
        Check molecular plausibility using RDKit.
        
        Returns list of warnings.
        """
        warnings_list = []
        
        # Check molecular weight
        mw = Descriptors.MolWt(mol)
        if mw > 2000:
            warnings_list.append(f"Very large molecule (MW={mw:.1f})")
        
        # Check number of atoms
        n_atoms = mol.GetNumAtoms()
        if n_atoms > 200:
            warnings_list.append(f"Many atoms (n={n_atoms})")
            
        # Check for radicals
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                warnings_list.append("Molecule contains radicals")
                break
                
        return warnings_list