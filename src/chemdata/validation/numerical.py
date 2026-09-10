"""
Numerical data validation.

Validates numerical ranges, units, and data types.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np


class NumericalValidator:
    """
    Validator for numerical data in chemistry datasets.
    
    Examples:
        >>> validator = NumericalValidator()
        >>> validator.validate_range(25.0, min_val=0, max_val=100, name="temperature")
        {'valid': True, 'value': 25.0, 'errors': []}
    """
    
    # Common chemistry value ranges
    STANDARD_RANGES = {
        'temperature_C': (-273.15, 1000),
        'temperature_K': (0, 1273),
        'pH': (0, 14),
        'concentration': (0, None),  # No upper limit
        'percentage': (0, 100),
        'degradation': (0, 100),
        'efficiency': (0, 100),
        'time': (0, None),
        'energy': (None, None),  # Can be negative
        'bandgap_eV': (0, 10),
        'surface_area': (0, None),
        'catalyst_loading': (0, None),
    }
    
    def __init__(self):
        """Initialize numerical validator."""
        pass
    
    def validate_range(
        self,
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "value",
        allow_nan: bool = False
    ) -> Dict:
        """
        Validate that value is within range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            name: Name of value for error messages
            allow_nan: Whether to allow NaN values
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'value': value,
            'errors': [],
            'warnings': []
        }
        
        # Check if NaN
        if pd.isna(value):
            if allow_nan:
                result['valid'] = True
                return result
            else:
                result['errors'].append(f"{name} is missing (NaN)")
                return result
        
        # Check if numeric
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            result['errors'].append(f"{name} is not numeric: {value}")
            return result
        
        result['value'] = numeric_value
        
        # Check range
        if min_val is not None and numeric_value < min_val:
            result['errors'].append(
                f"{name} = {numeric_value} is below minimum ({min_val})"
            )
            return result
            
        if max_val is not None and numeric_value > max_val:
            result['errors'].append(
                f"{name} = {numeric_value} exceeds maximum ({max_val})"
            )
            return result
        
        # Warnings for edge cases
        if min_val is not None and numeric_value == min_val:
            result['warnings'].append(f"{name} is at minimum value")
            
        if max_val is not None and numeric_value == max_val:
            result['warnings'].append(f"{name} is at maximum value")
        
        result['valid'] = True
        return result
    
    def validate_percentage(self, value: Any, name: str = "percentage") -> Dict:
        """Validate percentage value (0-100)."""
        return self.validate_range(value, min_val=0, max_val=100, name=name)
    
    def validate_positive(self, value: Any, name: str = "value") -> Dict:
        """Validate positive value (>0)."""
        return self.validate_range(value, min_val=0, max_val=None, name=name)
    
    def validate_ph(self, value: Any) -> Dict:
        """Validate pH value (0-14)."""
        return self.validate_range(value, min_val=0, max_val=14, name="pH")
    
    def validate_temperature_celsius(self, value: Any) -> Dict:
        """Validate temperature in Celsius."""
        return self.validate_range(
            value, 
            min_val=-273.15, 
            max_val=1000, 
            name="temperature (°C)"
        )
    
    def validate_column(
        self,
        df: pd.DataFrame,
        column: str,
        validator_type: str = 'positive',
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Validate entire column of DataFrame.
        
        Args:
            df: Input DataFrame
            column: Column to validate
            validator_type: Type of validation ('positive', 'percentage', 'pH', 'range')
            min_val: Minimum value (for 'range' type)
            max_val: Maximum value (for 'range' type)
            
        Returns:
            DataFrame with validation results added
        """
        # Select validator
        if validator_type == 'positive':
            validator_func = self.validate_positive
        elif validator_type == 'percentage':
            validator_func = self.validate_percentage
        elif validator_type == 'pH':
            validator_func = self.validate_ph
        elif validator_type == 'range':
            validator_func = lambda x: self.validate_range(x, min_val, max_val, column)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")
        
        # Apply validation
        validation_results = df[column].apply(validator_func)
        
        df[f'{column}_valid'] = validation_results.apply(lambda x: x['valid'])
        df[f'{column}_errors'] = validation_results.apply(
            lambda x: '; '.join(x['errors']) if x['errors'] else None
        )
        df[f'{column}_warnings'] = validation_results.apply(
            lambda x: '; '.join(x['warnings']) if x['warnings'] else None
        )
        
        return df
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Dict]
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate entire DataFrame against schema.
        
        Args:
            df: Input DataFrame
            schema: Dictionary mapping column names to validation specs
                   e.g., {'temperature': {'type': 'range', 'min': 0, 'max': 100}}
                   
        Returns:
            Tuple of (validated DataFrame, summary report)
        """
        validated_df = df.copy()
        report = {
            'total_rows': len(df),
            'columns_validated': 0,
            'validation_errors': {},
            'summary': {}
        }
        
        for column, spec in schema.items():
            if column not in df.columns:
                report['validation_errors'][column] = f"Column not found in DataFrame"
                continue
            
            validator_type = spec.get('type', 'positive')
            min_val = spec.get('min')
            max_val = spec.get('max')
            
            validated_df = self.validate_column(
                validated_df,
                column,
                validator_type=validator_type,
                min_val=min_val,
                max_val=max_val
            )
            
            # Summary statistics
            n_valid = validated_df[f'{column}_valid'].sum()
            n_invalid = (~validated_df[f'{column}_valid']).sum()
            
            report['summary'][column] = {
                'valid': int(n_valid),
                'invalid': int(n_invalid),
                'validity_rate': float(n_valid) / len(df) if len(df) > 0 else 0
            }
            
            report['columns_validated'] += 1
        
        return validated_df, report
    
    def check_units_consistency(
        self,
        df: pd.DataFrame,
        column: str,
        expected_range: Tuple[float, float],
        unit_name: str
    ) -> Dict:
        """
        Check if values are in expected range for given units.
        
        Useful for detecting unit mismatches (e.g., using mM when expecting mg/L).
        
        Args:
            df: Input DataFrame
            column: Column to check
            expected_range: Expected (min, max) range
            unit_name: Name of units for reporting
            
        Returns:
            Dictionary with check results
        """
        values = df[column].dropna()
        
        if len(values) == 0:
            return {
                'consistent': None,
                'message': 'No values to check'
            }
        
        min_expected, max_expected = expected_range
        min_actual = values.min()
        max_actual = values.max()
        
        # Check if values are in expected range
        values_in_range = (
            (min_expected <= min_actual) and
            (max_actual <= max_expected)
        )
        
        # Check for suspicious patterns
        warnings = []
        
        if min_actual < min_expected * 0.1:
            warnings.append(
                f"Minimum value ({min_actual:.2e}) much smaller than expected for {unit_name}"
            )
            
        if max_actual > max_expected * 10:
            warnings.append(
                f"Maximum value ({max_actual:.2e}) much larger than expected for {unit_name}"
            )
        
        return {
            'consistent': values_in_range,
            'min_actual': float(min_actual),
            'max_actual': float(max_actual),
            'min_expected': float(min_expected),
            'max_expected': float(max_expected),
            'warnings': warnings
        }