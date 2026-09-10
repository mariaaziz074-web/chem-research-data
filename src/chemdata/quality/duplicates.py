"""
Duplicate detection for chemistry datasets.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class DuplicateDetector:
    """
    Detect duplicate records in chemistry datasets.
    
    Handles:
    - Exact duplicates
    - Chemical duplicates (same molecule)
    - Experimental duplicates (same conditions)
    - Near-duplicates
    """
    
    def __init__(self):
        """Initialize duplicate detector."""
        self.rdkit_available = RDKIT_AVAILABLE
    
    def find_exact_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Find exact duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to check for duplicates (None = all columns)
            
        Returns:
            DataFrame with duplicate rows
        """
        duplicates = df[df.duplicated(subset=subset, keep=False)]
        return duplicates.sort_values(by=subset if subset else df.columns.tolist())
    
    def mark_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Mark duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to check
            keep: Which duplicates to mark ('first', 'last', False)
            
        Returns:
            DataFrame with 'is_duplicate' column added
        """
        df = df.copy()
        df['is_duplicate'] = df.duplicated(subset=subset, keep=keep)
        return df
    
    def find_chemical_duplicates(
        self,
        df: pd.DataFrame,
        smiles_column: str
    ) -> Dict:
        """
        Find duplicate molecules (same chemical structure).
        
        Args:
            df: Input DataFrame
            smiles_column: Column containing SMILES strings
            
        Returns:
            Dictionary of duplicate groups
        """
        if not self.rdkit_available:
            return {'error': 'RDKit not available'}
        
        # Canonicalize SMILES
        canonical_smiles = {}
        duplicates = {}
        
        for idx, smiles in df[smiles_column].items():
            if pd.isna(smiles):
                continue
            
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    continue
                
                canonical = Chem.MolToSmiles(mol, canonical=True)
                
                if canonical in canonical_smiles:
                    # Duplicate found
                    if canonical not in duplicates:
                        duplicates[canonical] = [canonical_smiles[canonical]]
                    duplicates[canonical].append(idx)
                else:
                    canonical_smiles[canonical] = idx
                    
            except Exception:
                continue
        
        return duplicates
    
    def find_experimental_duplicates(
        self,
        df: pd.DataFrame,
        condition_columns: List[str],
        tolerance: Optional[Dict[str, float]] = None
    ) -> List[List[int]]:
        """
        Find duplicate experimental conditions.
        
        Args:
            df: Input DataFrame
            condition_columns: Columns defining experimental conditions
            tolerance: Tolerance for numerical columns (e.g., {'temperature': 1.0})
            
        Returns:
            List of duplicate groups (list of row indices)
        """
        tolerance = tolerance or {}
        duplicates = []
        checked = set()
        
        for i in range(len(df)):
            if i in checked:
                continue
            
            group = [i]
            
            for j in range(i + 1, len(df)):
                if j in checked:
                    continue
                
                is_duplicate = True
                
                for col in condition_columns:
                    val_i = df.iloc[i][col]
                    val_j = df.iloc[j][col]
                    
                    # Handle missing values
                    if pd.isna(val_i) or pd.isna(val_j):
                        if not (pd.isna(val_i) and pd.isna(val_j)):
                            is_duplicate = False
                            break
                        continue
                    
                    # Numerical comparison with tolerance
                    if col in tolerance:
                        if abs(float(val_i) - float(val_j)) > tolerance[col]:
                            is_duplicate = False
                            break
                    else:
                        # Exact comparison
                        if val_i != val_j:
                            is_duplicate = False
                            break
                
                if is_duplicate:
                    group.append(j)
                    checked.add(j)
            
            if len(group) > 1:
                duplicates.append(group)
                checked.update(group)
        
        return duplicates
    
    def generate_duplicate_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive duplicate report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Report dictionary
        """
        report = {
            'total_rows': len(df),
            'exact_duplicates': 0,
            'duplicate_fraction': 0.0,
        }
        
        # Count exact duplicates
        n_duplicates = df.duplicated().sum()
        report['exact_duplicates'] = int(n_duplicates)
        report['duplicate_fraction'] = float(n_duplicates) / len(df) if len(df) > 0 else 0.0
        
        return report