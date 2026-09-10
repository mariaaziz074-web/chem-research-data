"""
Chemical name and formula normalization.
"""

from typing import Dict, List, Optional, Set
import re
import pandas as pd

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class ChemicalNormalizer:
    """
    Normalize chemical names and formulas.
    
    Examples:
        >>> normalizer = ChemicalNormalizer()
        >>> normalizer.normalize_formula("Ti O2")
        "TiO2"
        
        >>> normalizer.standardize_smiles("CCO")
        "CCO"
    """
    
    # Common chemical name synonyms
    CHEMICAL_SYNONYMS = {
        'titanium dioxide': 'TiO2',
        'titanium oxide': 'TiO2',
        'titania': 'TiO2',
        'zinc oxide': 'ZnO',
        'iron oxide': 'Fe2O3',
        'copper oxide': 'CuO',
        'methylene blue': 'Methylene Blue',
        'rhodamine b': 'Rhodamine B',
        'methyl orange': 'Methyl Orange',
    }
    
    def __init__(self):
        """Initialize chemical normalizer."""
        self.rdkit_available = RDKIT_AVAILABLE
    
    def normalize_formula(self, formula: str) -> str:
        """
        Normalize chemical formula.
        
        Args:
            formula: Chemical formula
            
        Returns:
            Normalized formula
        """
        if not isinstance(formula, str):
            return formula
        
        # Remove spaces
        normalized = formula.replace(' ', '')
        
        # Convert subscript numbers to normal
        subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        normalized = normalized.translate(subscript_map)
        
        # Convert superscript to normal
        superscript_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')
        normalized = normalized.translate(superscript_map)
        
        # Capitalize properly (Ti O2 -> TiO2)
        normalized = self._fix_capitalization(normalized)
        
        return normalized
    
    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        Standardize SMILES string using RDKit.
        
        Args:
            smiles: Input SMILES
            
        Returns:
            Canonical SMILES or None if invalid
        """
        if not self.rdkit_available:
            return smiles
        
        if not isinstance(smiles, str):
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    def normalize_chemical_name(self, name: str) -> str:
        """
        Normalize chemical name.
        
        Args:
            name: Chemical name
            
        Returns:
            Normalized name
        """
        if not isinstance(name, str):
            return name
        
        # Convert to lowercase for lookup
        name_lower = name.lower().strip()
        
        # Check synonyms
        if name_lower in self.CHEMICAL_SYNONYMS:
            return self.CHEMICAL_SYNONYMS[name_lower]
        
        # Basic normalization
        normalized = name.strip()
        
        # Standardize hyphenation
        normalized = normalized.replace('_', '-')
        
        return normalized
    
    def normalize_catalyst_name(self, catalyst: str) -> str:
        """
        Normalize catalyst name.
        
        Handles common variations like:
        - TiO2 / TiO₂ / Ti O2
        - P25 / Degussa P25
        
        Args:
            catalyst: Catalyst name or formula
            
        Returns:
            Normalized catalyst name
        """
        if not isinstance(catalyst, str):
            return catalyst
        
        catalyst = catalyst.strip()
        
        # Check if it's a formula
        if self._is_likely_formula(catalyst):
            return self.normalize_formula(catalyst)
        
        # Normalize common catalyst names
        catalyst_lower = catalyst.lower()
        
        if 'p25' in catalyst_lower or 'p-25' in catalyst_lower:
            return 'TiO2 (P25)'
        
        if 'degussa' in catalyst_lower:
            return 'TiO2 (Degussa)'
        
        return catalyst
    
    def normalize_dataframe_column(
        self,
        df: pd.DataFrame,
        column: str,
        normalization_type: str = 'formula',
        new_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Normalize chemistry column in DataFrame.
        
        Args:
            df: Input DataFrame
            column: Column to normalize
            normalization_type: 'formula', 'smiles', 'name', 'catalyst'
            new_column: Name for normalized column
            
        Returns:
            DataFrame with normalized column
        """
        if new_column is None:
            new_column = f"{column}_normalized"
        
        normalizers = {
            'formula': self.normalize_formula,
            'smiles': self.standardize_smiles,
            'name': self.normalize_chemical_name,
            'catalyst': self.normalize_catalyst_name,
        }
        
        if normalization_type not in normalizers:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
        
        normalizer = normalizers[normalization_type]
        df[new_column] = df[column].apply(normalizer)
        
        return df
    
    @staticmethod
    def _is_likely_formula(text: str) -> bool:
        """Check if text is likely a chemical formula."""
        # Formula pattern: starts with capital letter, contains numbers
        pattern = r'^[A-Z][A-Za-z0-9₀₁₂₃₄₅₆₇₈₉\(\)]+$'
        return bool(re.match(pattern, text))
    
    @staticmethod
    def _fix_capitalization(formula: str) -> str:
        """Fix element capitalization in formula."""
        # This is simplified - just ensures first letter is capital
        # More sophisticated version would parse elements
        if not formula:
            return formula
        
        # Don't change if already properly formatted
        if formula[0].isupper():
            return formula
        
        return formula[0].upper() + formula[1:]