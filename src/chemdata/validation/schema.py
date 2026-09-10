"""
Dataset schema validation.

Validates dataset structure, column types, and required fields.
"""

from typing import Dict, List, Optional, Any, Set
import pandas as pd
import numpy as np


class SchemaValidator:
    """
    Validator for dataset schemas.
    
    Examples:
        >>> schema = {
        ...     'catalyst': {'type': 'string', 'required': True},
        ...     'temperature': {'type': 'float', 'required': True, 'min': 0},
        ...     'degradation': {'type': 'float', 'required': True, 'min': 0, 'max': 100}
        ... }
        >>> validator = SchemaValidator(schema)
        >>> report = validator.validate(df)
    """
    
    VALID_TYPES = {
        'string', 'str', 'text',
        'int', 'integer',
        'float', 'number', 'numeric',
        'bool', 'boolean',
        'datetime', 'date',
        'category', 'categorical'
    }
    
    def __init__(self, schema: Optional[Dict] = None):
        """
        Initialize schema validator.
        
        Args:
            schema: Schema dictionary mapping column names to specifications
        """
        self.schema = schema or {}
        self._validate_schema()
    
    def _validate_schema(self):
        """Validate that schema itself is well-formed."""
        for column, spec in self.schema.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Schema for '{column}' must be dictionary")
            
            if 'type' in spec and spec['type'] not in self.VALID_TYPES:
                raise ValueError(
                    f"Invalid type '{spec['type']}' for column '{column}'. "
                    f"Must be one of: {self.VALID_TYPES}"
                )
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation report dictionary
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'column_reports': {},
            'summary': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'schema_columns': len(self.schema),
                'missing_required_columns': [],
                'extra_columns': [],
            }
        }
        
        # Check for required columns
        required_columns = {
            col for col, spec in self.schema.items() 
            if spec.get('required', False)
        }
        
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            report['valid'] = False
            report['errors'].append(
                f"Missing required columns: {missing_columns}"
            )
            report['summary']['missing_required_columns'] = list(missing_columns)
        
        # Check for extra columns
        extra_columns = set(df.columns) - set(self.schema.keys())
        if extra_columns:
            report['warnings'].append(
                f"Extra columns not in schema: {extra_columns}"
            )
            report['summary']['extra_columns'] = list(extra_columns)
        
        # Validate each column
        for column, spec in self.schema.items():
            if column not in df.columns:
                continue  # Already reported as missing
            
            column_report = self._validate_column(df[column], column, spec)
            report['column_reports'][column] = column_report
            
            if not column_report['valid']:
                report['valid'] = False
                report['errors'].extend(column_report['errors'])
            
            report['warnings'].extend(column_report['warnings'])
        
        return report
    
    def _validate_column(
        self,
        series: pd.Series,
        column_name: str,
        spec: Dict
    ) -> Dict:
        """
        Validate single column against specification.
        
        Args:
            series: Column data
            column_name: Name of column
            spec: Column specification
            
        Returns:
            Column validation report
        """
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check data type
        expected_type = spec.get('type')
        if expected_type:
            type_valid, type_msg = self._check_type(series, expected_type)
            if not type_valid:
                report['valid'] = False
                report['errors'].append(f"{column_name}: {type_msg}")
        
        # Check for missing values
        n_missing = series.isna().sum()
        allow_missing = spec.get('allow_missing', True)
        
        report['stats']['n_missing'] = int(n_missing)
        report['stats']['missing_fraction'] = float(n_missing) / len(series)
        
        if n_missing > 0 and not allow_missing:
            report['valid'] = False
            report['errors'].append(
                f"{column_name}: Contains {n_missing} missing values but none allowed"
            )
        
        # Check unique constraint
        if spec.get('unique', False):
            n_unique = series.nunique()
            if n_unique < len(series.dropna()):
                report['valid'] = False
                report['errors'].append(
                    f"{column_name}: Must be unique but has duplicates"
                )
        
        # Check allowed values
        allowed_values = spec.get('allowed_values')
        if allowed_values is not None:
            invalid_values = set(series.dropna().unique()) - set(allowed_values)
            if invalid_values:
                report['valid'] = False
                report['errors'].append(
                    f"{column_name}: Contains invalid values: {invalid_values}"
                )
        
        # Check numeric ranges
        if expected_type in {'float', 'number', 'numeric', 'int', 'integer'}:
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            min_val = spec.get('min')
            if min_val is not None:
                below_min = (numeric_series < min_val).sum()
                if below_min > 0:
                    report['valid'] = False
                    report['errors'].append(
                        f"{column_name}: {below_min} values below minimum ({min_val})"
                    )
            
            max_val = spec.get('max')
            if max_val is not None:
                above_max = (numeric_series > max_val).sum()
                if above_max > 0:
                    report['valid'] = False
                    report['errors'].append(
                        f"{column_name}: {above_max} values above maximum ({max_val})"
                    )
            
            # Add statistics
            report['stats']['mean'] = float(numeric_series.mean())
            report['stats']['std'] = float(numeric_series.std())
            report['stats']['min'] = float(numeric_series.min())
            report['stats']['max'] = float(numeric_series.max())
        
        return report
    
    @staticmethod
    def _check_type(series: pd.Series, expected_type: str) -> tuple:
        """
        Check if series matches expected type.
        
        Returns:
            (is_valid, message)
        """
        # Map expected types to pandas dtypes
        if expected_type in {'string', 'str', 'text'}:
            if not pd.api.types.is_string_dtype(series):
                # Check if can be converted
                try:
                    series.astype(str)
                    return True, "Can be converted to string"
                except:
                    return False, "Cannot convert to string"
            return True, "Valid string type"
        
        elif expected_type in {'int', 'integer'}:
            if not pd.api.types.is_integer_dtype(series):
                # Check if can be converted (allowing for NaN)
                if pd.api.types.is_numeric_dtype(series):
                    return True, "Numeric type compatible with integer"
                return False, "Not integer type"
            return True, "Valid integer type"
        
        elif expected_type in {'float', 'number', 'numeric'}:
            if not pd.api.types.is_numeric_dtype(series):
                # Try to convert
                try:
                    pd.to_numeric(series, errors='coerce')
                    return True, "Can be converted to numeric"
                except:
                    return False, "Cannot convert to numeric"
            return True, "Valid numeric type"
        
        elif expected_type in {'bool', 'boolean'}:
            if not pd.api.types.is_bool_dtype(series):
                unique_values = set(series.dropna().unique())
                bool_values = {True, False, 'True', 'False', 'true', 'false', 1, 0}
                if not unique_values.issubset(bool_values):
                    return False, "Not boolean type"
            return True, "Valid boolean type"
        
        elif expected_type in {'datetime', 'date'}:
            if not pd.api.types.is_datetime64_any_dtype(series):
                try:
                    pd.to_datetime(series, errors='coerce')
                    return True, "Can be converted to datetime"
                except:
                    return False, "Cannot convert to datetime"
            return True, "Valid datetime type"
        
        elif expected_type in {'category', 'categorical'}:
            return True, "Categorical type check passed"
        
        return True, "Type check passed"
    
    def generate_schema_from_dataframe(
        self,
        df: pd.DataFrame,
        mark_all_required: bool = False
    ) -> Dict:
        """
        Generate schema from existing DataFrame.
        
        Useful for creating initial schema templates.
        
        Args:
            df: Input DataFrame
            mark_all_required: Whether to mark all columns as required
            
        Returns:
            Generated schema dictionary
        """
        schema = {}
        
        for column in df.columns:
            dtype = df[column].dtype
            
            spec = {
                'required': mark_all_required,
                'allow_missing': df[column].isna().any(),
            }
            
            # Infer type
            if pd.api.types.is_string_dtype(dtype):
                spec['type'] = 'string'
            elif pd.api.types.is_integer_dtype(dtype):
                spec['type'] = 'integer'
                spec['min'] = int(df[column].min())
                spec['max'] = int(df[column].max())
            elif pd.api.types.is_float_dtype(dtype):
                spec['type'] = 'float'
                spec['min'] = float(df[column].min())
                spec['max'] = float(df[column].max())
            elif pd.api.types.is_bool_dtype(dtype):
                spec['type'] = 'boolean'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                spec['type'] = 'datetime'
            else:
                spec['type'] = 'string'  # Default
            
            schema[column] = spec
        
        return schema