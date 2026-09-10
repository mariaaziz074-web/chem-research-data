"""
Unit normalization and conversion.

Handles conversion between different unit systems for:
- Concentration (mg/L, mM, ppm, μg/mL)
- Energy (eV, kcal/mol, kJ/mol, Hartree)
- Temperature (°C, K, °F)
- Time (s, min, h, day)
"""

from typing import Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pint import UnitRegistry

# Initialize unit registry
ureg = UnitRegistry()


class UnitNormalizer:
    """
    Normalize and convert units for chemistry data.
    
    Examples:
        >>> normalizer = UnitNormalizer()
        >>> normalizer.convert_concentration(10, 'mg/L', 'mM', molecular_weight=180)
        0.0556  # mM
        
        >>> normalizer.convert_energy(1, 'eV', 'kcal/mol')
        23.06  # kcal/mol
    """
    
    # Conversion factors
    ENERGY_CONVERSIONS = {
        ('eV', 'kcal/mol'): 23.06054783,
        ('eV', 'kJ/mol'): 96.48533213,
        ('eV', 'Hartree'): 0.03674932,
        ('kcal/mol', 'kJ/mol'): 4.184,
        ('kcal/mol', 'eV'): 1/23.06054783,
        ('kJ/mol', 'kcal/mol'): 1/4.184,
        ('kJ/mol', 'eV'): 1/96.48533213,
        ('Hartree', 'eV'): 27.211386245,
    }
    
    def __init__(self):
        """Initialize unit normalizer."""
        self.ureg = ureg
    
    def convert_temperature(
        self, 
        value: float, 
        from_unit: str, 
        to_unit: str
    ) -> float:
        """
        Convert temperature between units.
        
        Args:
            value: Temperature value
            from_unit: Source unit ('C', 'K', 'F')
            to_unit: Target unit ('C', 'K', 'F')
            
        Returns:
            Converted temperature
        """
        # Normalize unit names
        unit_map = {
            'C': 'degC', 'celsius': 'degC', '°C': 'degC',
            'K': 'kelvin', 'kelvin': 'kelvin',
            'F': 'degF', 'fahrenheit': 'degF', '°F': 'degF'
        }
        
        from_unit = unit_map.get(from_unit, from_unit)
        to_unit = unit_map.get(to_unit, to_unit)
        
        # Convert using pint
        quantity = self.ureg.Quantity(value, from_unit)
        converted = quantity.to(to_unit)
        
        return converted.magnitude
    
    def convert_time(
        self, 
        value: float, 
        from_unit: str, 
        to_unit: str
    ) -> float:
        """
        Convert time between units.
        
        Args:
            value: Time value
            from_unit: Source unit ('s', 'min', 'h', 'day')
            to_unit: Target unit
            
        Returns:
            Converted time
        """
        # Normalize unit names
        unit_map = {
            's': 'second', 'sec': 'second', 'second': 'second',
            'min': 'minute', 'minute': 'minute',
            'h': 'hour', 'hr': 'hour', 'hour': 'hour',
            'd': 'day', 'day': 'day'
        }
        
        from_unit = unit_map.get(from_unit, from_unit)
        to_unit = unit_map.get(to_unit, to_unit)
        
        quantity = self.ureg.Quantity(value, from_unit)
        converted = quantity.to(to_unit)
        
        return converted.magnitude
    
    def convert_energy(
        self, 
        value: float, 
        from_unit: str, 
        to_unit: str
    ) -> float:
        """
        Convert energy between units.
        
        Args:
            value: Energy value
            from_unit: Source unit ('eV', 'kcal/mol', 'kJ/mol', 'Hartree')
            to_unit: Target unit
            
        Returns:
            Converted energy
        """
        if from_unit == to_unit:
            return value
        
        # Check if we have direct conversion
        key = (from_unit, to_unit)
        if key in self.ENERGY_CONVERSIONS:
            return value * self.ENERGY_CONVERSIONS[key]
        
        # Try reverse conversion
        reverse_key = (to_unit, from_unit)
        if reverse_key in self.ENERGY_CONVERSIONS:
            return value / self.ENERGY_CONVERSIONS[reverse_key]
        
        # Multi-step conversion via eV
        if from_unit != 'eV':
            value_eV = self.convert_energy(value, from_unit, 'eV')
        else:
            value_eV = value
            
        if to_unit != 'eV':
            return self.convert_energy(value_eV, 'eV', to_unit)
        else:
            return value_eV
    
    def convert_concentration(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        molecular_weight: Optional[float] = None,
        volume_L: float = 1.0
    ) -> float:
        """
        Convert concentration between units.
        
        Args:
            value: Concentration value
            from_unit: Source unit ('mg/L', 'mM', 'ppm', 'μg/mL', 'M')
            to_unit: Target unit
            molecular_weight: Molecular weight in g/mol (required for molar conversions)
            volume_L: Volume in liters (default: 1.0)
            
        Returns:
            Converted concentration
        """
        if from_unit == to_unit:
            return value
        
        # Normalize unit names
        unit_aliases = {
            'mg/L': 'mg/L',
            'mg/l': 'mg/L',
            'ppm': 'mg/L',  # For aqueous solutions
            'μg/mL': 'mg/L',
            'ug/mL': 'mg/L',
            'g/L': 'g/L',
            'mM': 'mM',
            'mm': 'mM',
            'μM': 'μM',
            'uM': 'μM',
            'M': 'M',
        }
        
        from_unit = unit_aliases.get(from_unit, from_unit)
        to_unit = unit_aliases.get(to_unit, to_unit)
        
        # Convert to mg/L as intermediate
        if from_unit == 'mg/L':
            mg_per_L = value
        elif from_unit == 'g/L':
            mg_per_L = value * 1000
        elif from_unit in ['mM', 'μM', 'M']:
            if molecular_weight is None:
                raise ValueError("Molecular weight required for molar concentration conversion")
            
            if from_unit == 'M':
                molar = value
            elif from_unit == 'mM':
                molar = value / 1000
            elif from_unit == 'μM':
                molar = value / 1_000_000
            
            mg_per_L = molar * molecular_weight * 1000  # mol/L * g/mol * 1000 mg/g
        else:
            raise ValueError(f"Unknown unit: {from_unit}")
        
        # Convert from mg/L to target
        if to_unit == 'mg/L':
            return mg_per_L
        elif to_unit == 'g/L':
            return mg_per_L / 1000
        elif to_unit in ['mM', 'μM', 'M']:
            if molecular_weight is None:
                raise ValueError("Molecular weight required for molar concentration conversion")
            
            molar = (mg_per_L / 1000) / molecular_weight  # mg/L / 1000 / (g/mol) = mol/L
            
            if to_unit == 'M':
                return molar
            elif to_unit == 'mM':
                return molar * 1000
            elif to_unit == 'μM':
                return molar * 1_000_000
        else:
            raise ValueError(f"Unknown unit: {to_unit}")
    
    def normalize_column(
        self,
        df: pd.DataFrame,
        column: str,
        from_unit: str,
        to_unit: str,
        conversion_type: str,
        molecular_weight: Optional[float] = None,
        new_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Normalize entire column to new units.
        
        Args:
            df: Input DataFrame
            column: Column to normalize
            from_unit: Source unit
            to_unit: Target unit
            conversion_type: 'temperature', 'time', 'energy', or 'concentration'
            molecular_weight: MW for concentration conversions
            new_column: Name for new column (default: column_<to_unit>)
            
        Returns:
            DataFrame with normalized column added
        """
        if new_column is None:
            new_column = f"{column}_{to_unit.replace('/', '_')}"
        
        conversion_functions = {
            'temperature': self.convert_temperature,
            'time': self.convert_time,
            'energy': self.convert_energy,
            'concentration': self.convert_concentration,
        }
        
        if conversion_type not in conversion_functions:
            raise ValueError(f"Unknown conversion type: {conversion_type}")
        
        converter = conversion_functions[conversion_type]
        
        # Apply conversion
        if conversion_type == 'concentration':
            df[new_column] = df[column].apply(
                lambda x: converter(x, from_unit, to_unit, molecular_weight)
                if pd.notna(x) else np.nan
            )
        else:
            df[new_column] = df[column].apply(
                lambda x: converter(x, from_unit, to_unit)
                if pd.notna(x) else np.nan
            )
        
        return df
    
    def detect_likely_unit(
        self,
        values: pd.Series,
        quantity_type: str
    ) -> Dict[str, float]:
        """
        Detect most likely unit based on value ranges.
        
        Useful for datasets with unspecified units.
        
        Args:
            values: Series of values
            quantity_type: 'temperature', 'time', 'energy', 'concentration'
            
        Returns:
            Dictionary of likely units with confidence scores
        """
        values_clean = values.dropna()
        
        if len(values_clean) == 0:
            return {}
        
        min_val = values_clean.min()
        max_val = values_clean.max()
        mean_val = values_clean.mean()
        
        scores = {}
        
        if quantity_type == 'temperature':
            # Celsius: typically 0-100
            if 0 <= mean_val <= 100:
                scores['C'] = 0.8
            # Kelvin: typically 273-373
            if 200 <= mean_val <= 400:
                scores['K'] = 0.8
            # Fahrenheit: typically 32-212
            if 30 <= mean_val <= 220:
                scores['F'] = 0.5
                
        elif quantity_type == 'time':
            # Seconds: can be any range
            scores['s'] = 0.3
            # Minutes: typically 0-1000
            if 1 <= mean_val <= 500:
                scores['min'] = 0.7
            # Hours: typically 0-100
            if 0.1 <= mean_val <= 50:
                scores['h'] = 0.6
                
        elif quantity_type == 'energy':
            # eV: typically 0-10
            if 0 <= mean_val <= 20:
                scores['eV'] = 0.7
            # kcal/mol: typically 0-200
            if 0 <= mean_val <= 300:
                scores['kcal/mol'] = 0.6
            # kJ/mol: typically 0-800
            if 0 <= mean_val <= 1000:
                scores['kJ/mol'] = 0.6
        
        return scores