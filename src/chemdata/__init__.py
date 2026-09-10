"""
chemdata: Chemistry Dataset Quality Assurance Toolkit

A Python package for validating, normalizing, and quality-checking chemistry datasets
for machine learning and computational research.

Modules:
    validation: Validate chemical structures, formulas, and numerical data
    normalization: Normalize units, chemicals, and experimental conditions
    provenance: Track data sources and transformations
    quality: Detect duplicates, outliers, and missing data
    export: Export to ML-ready formats
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main classes for convenience
from chemdata.validation.chemical import ChemicalValidator
from chemdata.validation.numerical import NumericalValidator
from chemdata.validation.schema import SchemaValidator

__all__ = [
    "ChemicalValidator",
    "NumericalValidator",
    "SchemaValidator",
    "__version__",
]